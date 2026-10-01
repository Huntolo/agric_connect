"""
AgriConnect  — Farmer-to-Customer Marketplace with Powered Search
(FastAPI + SQLAlchemy + LLM query interpretation + RAG comparison pipeline)

This merges two things into one app:
  1. The original AgriConnect farmer-to-customer marketplace (auth, farmer
     profiles, product listings, cart/orders, payments, reviews, and the
     return/refund policy workflow).
  2. The coursework requirements, re-scoped to this domain: instead of
     comparing prices across external retailers (Jumia/Konga/Amazon), the
     platform uses an LLM to interpret natural-language customer queries and
     a RAG pipeline that retrieves REAL, live listings from AgriConnect's own
     farmers, compares them, and generates an AI summary recommending the
     best match(es).

     Example: "cheap organic tomatoes near me under 2000"
       -> LLM interprets: keywords="tomatoes", organic=True, max_price=2000
       -> RAG retrieval: query the Product table across all active farmers
       -> Comparison engine: rank by price + farmer rating
       -> LLM generation: "Farmer Ade's organic tomatoes at ₦1,800/kg offer
          the best value and have a 4.8 rating, though Farmer Musa's stock
          is slightly cheaper per kg if freshness matters less..."

Install:
    pip install fastapi uvicorn[standard] sqlalchemy pydantic pydantic-settings \
                python-jose[cryptography] passlib[bcrypt] python-multipart openai

Run:
    uvicorn agriconnect_ai_app:app --reload
    -> docs at http://localhost:8000/docs
"""

import enum
import json
import re
import uuid
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from pydantic_settings import BaseSettings
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, ForeignKey, DateTime, Enum, Text,
    create_engine, func,
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base, Session


# ======================================================================
# 1. CONFIG
# ======================================================================

class Settings(BaseSettings):
    app_name: str = "AgriConnect AI API"
    database_url: str = "sqlite:///./agriconnect.db"
    jwt_secret: str = "CHANGE_ME_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    platform_commission_percent: float = 8.0
    refund_report_window_hours: int = 24

    # LLM provider config — set via environment variables in production
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"


settings = Settings()


# ======================================================================
# 2. DATABASE
# ======================================================================

connect_args = {"check_same_thread": False} if "sqlite" in settings.database_url else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def gen_id() -> str:
    return str(uuid.uuid4())


# ======================================================================
# 3. MODELS
# ======================================================================

class UserRole(str, enum.Enum):
    FARMER = "farmer"
    CUSTOMER = "customer"
    ADMIN = "admin"
    DELIVERY_AGENT = "delivery_agent"


class OrderStatus(str, enum.Enum):
    PLACED = "placed"
    CONFIRMED = "confirmed"
    PREPARED = "prepared"
    OUT_FOR_DELIVERY = "out_for_delivery"
    READY_FOR_PICKUP = "ready_for_pickup"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DISPUTED = "disputed"


class PaymentStatus(str, enum.Enum):
    PENDING = "pending"
    PAID = "paid"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_id)
    name = Column(String, nullable=False)
    phone = Column(String, unique=True, index=True, nullable=True)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), nullable=False)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    farmer_profile = relationship("FarmerProfile", back_populates="user", uselist=False)
    favorites = relationship("Favorite", back_populates="user")


class FarmerProfile(Base):
    __tablename__ = "farmer_profiles"

    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=False)
    farm_name = Column(String, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    certifications = Column(String, nullable=True)  # comma-separated tags
    delivery_radius_km = Column(Float, default=10.0)

    user = relationship("User", back_populates="farmer_profile")
    products = relationship("Product", back_populates="farmer")


class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True, default=gen_id)
    farmer_id = Column(String, ForeignKey("farmer_profiles.id"), nullable=False)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    unit = Column(String, nullable=False)  # kg, dozen, crate, etc.
    price = Column(Float, nullable=False)
    stock_quantity = Column(Float, nullable=False, default=0)
    is_organic = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    farmer = relationship("FarmerProfile", back_populates="products")


class Order(Base):
    __tablename__ = "orders"

    id = Column(String, primary_key=True, default=gen_id)
    customer_id = Column(String, ForeignKey("users.id"), nullable=False)
    farmer_id = Column(String, ForeignKey("farmer_profiles.id"), nullable=False)
    status = Column(Enum(OrderStatus), default=OrderStatus.PLACED)
    fulfillment_type = Column(String, nullable=False)  # "delivery" | "pickup"
    delivery_fee = Column(Float, default=0.0)
    subtotal = Column(Float, nullable=False, default=0.0)
    total_amount = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)

    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")
    payment = relationship("Payment", back_populates="order", uselist=False)


class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(String, primary_key=True, default=gen_id)
    order_id = Column(String, ForeignKey("orders.id"), nullable=False)
    product_id = Column(String, ForeignKey("products.id"), nullable=False)
    quantity = Column(Float, nullable=False)
    unit_price = Column(Float, nullable=False)

    order = relationship("Order", back_populates="items")


class Payment(Base):
    __tablename__ = "payments"

    id = Column(String, primary_key=True, default=gen_id)
    order_id = Column(String, ForeignKey("orders.id"), unique=True, nullable=False)
    amount = Column(Float, nullable=False)
    method = Column(String, nullable=False)  # card, wallet, cod, mobile_money
    status = Column(Enum(PaymentStatus), default=PaymentStatus.PENDING)
    transaction_ref = Column(String, nullable=True)
    refunded_amount = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    order = relationship("Order", back_populates="payment")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(String, primary_key=True, default=gen_id)
    order_id = Column(String, ForeignKey("orders.id"), nullable=False)
    reviewer_id = Column(String, ForeignKey("users.id"), nullable=False)
    farmer_id = Column(String, ForeignKey("farmer_profiles.id"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class RefundRequest(Base):
    __tablename__ = "refund_requests"

    id = Column(String, primary_key=True, default=gen_id)
    order_id = Column(String, ForeignKey("orders.id"), nullable=False)
    reason = Column(String, nullable=False)
    evidence_url = Column(String, nullable=True)
    approved = Column(Boolean, nullable=True)  # None = pending
    resolved_by_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Favorite(Base):
    __tablename__ = "favorites"

    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    product_id = Column(String, ForeignKey("products.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="favorites")


# ======================================================================
# 4. SCHEMAS
# ======================================================================

class UserCreate(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    password: str = Field(min_length=8)
    role: UserRole


class UserOut(BaseModel):
    id: str
    name: str
    role: UserRole
    is_verified: bool

    class Config:
        from_attributes = True


class ProductCreate(BaseModel):
    name: str
    category: str
    description: Optional[str] = None
    unit: str
    price: float = Field(gt=0)
    stock_quantity: float = Field(ge=0)
    is_organic: bool = False


class ProductOut(ProductCreate):
    id: str
    farmer_id: str
    is_active: bool

    class Config:
        from_attributes = True


class OrderItemCreate(BaseModel):
    product_id: str
    quantity: float = Field(gt=0)


class OrderCreate(BaseModel):
    farmer_id: str
    fulfillment_type: str  # "delivery" | "pickup"
    items: List[OrderItemCreate]


class OrderOut(BaseModel):
    id: str
    status: OrderStatus
    subtotal: float
    delivery_fee: float
    total_amount: float
    created_at: datetime

    class Config:
        from_attributes = True


class PaymentCreate(BaseModel):
    order_id: str
    method: str  # card | wallet | cod | mobile_money


class PaymentOut(BaseModel):
    id: str
    order_id: str
    amount: float
    status: PaymentStatus

    class Config:
        from_attributes = True


class ReviewCreate(BaseModel):
    order_id: str
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None


class RefundRequestCreate(BaseModel):
    order_id: str
    reason: str
    evidence_url: Optional[str] = None


class FavoriteCreate(BaseModel):
    product_id: str


class SearchQuery(BaseModel):
    query: str = Field(..., description="Natural-language product search, e.g. "
                                         "'cheap organic tomatoes under 2000'")
    max_results: int = 10


class ProductMatch(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    unit: str
    is_organic: bool
    farmer_id: str
    farmer_name: str
    farmer_rating: Optional[float] = None

    class Config:
        from_attributes = True


# ======================================================================
# 5. AUTH
# ======================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/users/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(subject: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    return jwt.encode({"sub": subject, "exp": expire}, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    error = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise error
    except JWTError:
        raise error
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise error
    return user


# ======================================================================
# 6. MARKETPLACE SERVICES (orders, payouts, refunds)
# ======================================================================

def create_order(db: Session, customer_id: str, farmer_id: str,
                  fulfillment_type: str, items_in: list) -> Order:
    order = Order(
        customer_id=customer_id,
        farmer_id=farmer_id,
        fulfillment_type=fulfillment_type,
        status=OrderStatus.PLACED,
    )
    subtotal = 0.0

    for item in items_in:
        product = db.query(Product).filter(Product.id == item.product_id).first()
        if not product or not product.is_active:
            raise ValueError(f"Product {item.product_id} not available")
        if product.stock_quantity < item.quantity:
            raise ValueError(f"Insufficient stock for {product.name}")

        product.stock_quantity -= item.quantity
        if product.stock_quantity <= 0:
            product.is_active = False

        line_total = product.price * item.quantity
        subtotal += line_total
        order.items.append(OrderItem(
            product_id=product.id,
            quantity=item.quantity,
            unit_price=product.price,
        ))

    delivery_fee = 3.0 if fulfillment_type == "delivery" else 0.0
    order.subtotal = subtotal
    order.delivery_fee = delivery_fee
    order.total_amount = subtotal + delivery_fee

    db.add(order)
    db.commit()
    db.refresh(order)
    return order


def farmer_payout_amount(order_total: float) -> float:
    """Amount paid out to the farmer after platform commission."""
    commission = order_total * (settings.platform_commission_percent / 100)
    return round(order_total - commission, 2)


def is_within_report_window(order: Order) -> bool:
    """Return/refund policy: issues must be reported within the configured
    window (default 24h) after delivery/pickup."""
    if not order.delivered_at:
        return True
    deadline = order.delivered_at + timedelta(hours=settings.refund_report_window_hours)
    return datetime.utcnow() <= deadline


def file_refund_request(db: Session, order: Order, reason: str,
                         evidence_url: Optional[str]) -> RefundRequest:
    if not is_within_report_window(order):
        raise ValueError("Refund reporting window has expired")

    request = RefundRequest(order_id=order.id, reason=reason, evidence_url=evidence_url)
    order.status = OrderStatus.DISPUTED
    db.add(request)
    db.commit()
    db.refresh(request)
    return request


def approve_refund(db: Session, request: RefundRequest, full: bool = True) -> Payment:
    order = db.query(Order).filter(Order.id == request.order_id).first()
    payment = db.query(Payment).filter(Payment.order_id == order.id).first()
    if not payment or payment.status != PaymentStatus.PAID:
        raise ValueError("No captured payment found for this order")

    refund_amount = payment.amount if full else payment.amount / 2
    payment.refunded_amount = refund_amount
    payment.status = (
        PaymentStatus.REFUNDED if refund_amount >= payment.amount
        else PaymentStatus.PARTIALLY_REFUNDED
    )
    request.approved = True
    request.resolved_by_admin = True
    order.status = OrderStatus.CANCELLED

    db.commit()
    db.refresh(payment)
    return payment


# ======================================================================
# 7. LLM SERVICE  (query interpretation + generation half of RAG)
# ======================================================================

class LLMService:
    """Thin wrapper so the OpenAI client is easy to swap for Claude/Gemini.
    Falls back to a rule-based stub when no API key is configured, so the
    pipeline still runs end-to-end in development."""

    def __init__(self):
        self.enabled = bool(settings.openai_api_key)
        if self.enabled:
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.openai_api_key)

    def interpret_query(self, query: str) -> Dict[str, Any]:
        """Extract structured search filters from a natural-language query."""
        if not self.enabled:
            return self._stub_interpret(query)

        prompt = (
            "Extract farm-produce search parameters from the user's request. "
            "Return strict JSON with keys: keywords (string), max_price "
            "(number or null), organic_only (boolean), category (string or null).\n\n"
            f"User request: {query}"
        )
        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def summarize_comparison(self, query: str, matches: List[ProductMatch]) -> str:
        """Augmented generation step: feed retrieved, ranked listings to the
        LLM so it produces a grounded comparison across farmers."""
        if not self.enabled:
            return self._stub_summarize(query, matches)

        context = "\n".join(
            f"- {m.farmer_name}: {m.name} ({m.unit}) — ₦{m.price:,.0f}"
            f"{' [organic]' if m.is_organic else ''}, "
            f"farmer rating {m.farmer_rating or 'n/a'}"
            for m in matches[:10]
        )
        prompt = (
            f"Customer asked: \"{query}\"\n\n"
            f"Matching listings from AgriConnect farmers:\n{context}\n\n"
            "Write a concise, helpful comparison (3-5 sentences) recommending "
            "the best option(s) and explaining the trade-offs (price vs. "
            "rating vs. certification). Only use the data provided above."
        )
        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    # ---- fallbacks used when no LLM API key is configured ----
    def _stub_interpret(self, query: str) -> Dict[str, Any]:
        price_match = re.search(r"(\d[\d,]{2,})", query.replace(",", ""))
        max_price = float(price_match.group(1)) if price_match else None
        organic_only = "organic" in query.lower()
        cleaned = re.sub(r"under.*|below.*", "", query, flags=re.IGNORECASE)
        cleaned = re.sub(r"\borganic\b|\bcheap\b", "", cleaned, flags=re.IGNORECASE).strip()
        return {
            "keywords": cleaned or query,
            "max_price": max_price,
            "organic_only": organic_only,
            "category": None,
        }

    def _stub_summarize(self, query: str, matches: List[ProductMatch]) -> str:
        if not matches:
            return "No matching listings were found from active farmers for this search."
        best = matches[0]
        return (
            f"Based on {len(matches)} matching listing(s) for \"{query}\", the best value is "
            f"{best.name} from {best.farmer_name} at ₦{best.price:,.0f} per {best.unit}"
            f"{' (organic)' if best.is_organic else ''}, rated "
            f"{best.farmer_rating or 'n/a'}/5. Other farmers had similar produce at "
            f"comparable or higher prices — see the full list for alternatives."
        )


llm_service = LLMService()


# ======================================================================
# 8. RAG SEARCH PIPELINE (retrieval = AgriConnect's own Product table)
# ======================================================================

def retrieve_matching_products(db: Session, keywords: str, max_price: Optional[float],
                                organic_only: bool, category: Optional[str],
                                limit: int) -> List[ProductMatch]:
    query = db.query(Product, FarmerProfile, User).join(
        FarmerProfile, Product.farmer_id == FarmerProfile.id
    ).join(
        User, FarmerProfile.user_id == User.id
    ).filter(Product.is_active == True)  # noqa: E712

    if keywords:
        like = f"%{keywords.strip()}%"
        query = query.filter(
            (Product.name.ilike(like)) |
            (Product.description.ilike(like)) |
            (Product.category.ilike(like))
        )
    if category:
        query = query.filter(Product.category.ilike(f"%{category}%"))
    if organic_only:
        query = query.filter(Product.is_organic == True)  # noqa: E712
    if max_price is not None:
        query = query.filter(Product.price <= max_price)

    rows = query.limit(limit * 3).all()  # over-fetch, then rank below

    matches = []
    for product, farmer_profile, farmer_user in rows:
        avg_rating = db.query(func.avg(Review.rating)).filter(
            Review.farmer_id == farmer_profile.id
        ).scalar()
        matches.append(ProductMatch(
            product_id=product.id,
            name=product.name,
            category=product.category,
            price=product.price,
            unit=product.unit,
            is_organic=product.is_organic,
            farmer_id=farmer_profile.id,
            farmer_name=farmer_profile.farm_name,
            farmer_rating=round(avg_rating, 1) if avg_rating else None,
        ))
    return matches


def rank_matches(matches: List[ProductMatch]) -> List[ProductMatch]:
    def score(m: ProductMatch) -> float:
        rating_component = (m.farmer_rating or 3.0) / 5.0
        price_component = 1 / (1 + m.price / 1000)
        return 0.6 * price_component + 0.4 * rating_component

    return sorted(matches, key=score, reverse=True)


def run_ai_search(db: Session, user_query: str, max_results: int = 10) -> Dict[str, Any]:
    # Step 1 — interpret the natural-language query (LLM)
    parsed = llm_service.interpret_query(user_query)

    # Step 2 — retrieval: query AgriConnect's own live product listings
    matches = retrieve_matching_products(
        db,
        keywords=parsed.get("keywords", user_query),
        max_price=parsed.get("max_price"),
        organic_only=parsed.get("organic_only", False),
        category=parsed.get("category"),
        limit=max_results,
    )

    # Step 3 — price comparison engine
    ranked = rank_matches(matches)[:max_results]

    # Step 4 — augmented generation: summarize using retrieved, ranked data
    summary = llm_service.summarize_comparison(user_query, ranked)

    return {
        "interpreted_query": parsed,
        "results": ranked,
        "ai_summary": summary,
    }


# ======================================================================
# 9. ROUTERS
# ======================================================================

# ---- Users --------------------------------------------------------------
users_router = APIRouter(prefix="/api/v1/users", tags=["users"])


@users_router.post("/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(
        (User.email == payload.email) | (User.phone == payload.phone)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    user = User(
        name=payload.name,
        phone=payload.phone,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@users_router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(
        (User.email == form_data.username) | (User.phone == form_data.username)
    ).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect credentials")

    token = create_access_token(subject=user.id)
    return {"access_token": token, "token_type": "bearer"}


# ---- Products ------------------------------------------------------------
products_router = APIRouter(prefix="/api/v1/products", tags=["products"])


@products_router.post("/", response_model=ProductOut)
def create_product(payload: ProductCreate, db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    if current_user.role != UserRole.FARMER:
        raise HTTPException(status_code=403, detail="Only farmers can create listings")

    profile = db.query(FarmerProfile).filter(FarmerProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=400, detail="Complete your farmer profile first")

    product = Product(farmer_id=profile.id, **payload.model_dump())
    db.add(product)
    db.commit()
    db.refresh(product)
    return product


@products_router.get("/", response_model=List[ProductOut])
def list_products(category: Optional[str] = None, organic_only: bool = False,
                   db: Session = Depends(get_db)):
    query = db.query(Product).filter(Product.is_active == True)  # noqa: E712
    if category:
        query = query.filter(Product.category == category)
    if organic_only:
        query = query.filter(Product.is_organic == True)  # noqa: E712
    return query.all()


# ---- AI Search (LLM + RAG over AgriConnect's own listings) --------------
search_router = APIRouter(prefix="/api/v1/search", tags=["ai-search"])


@search_router.post("/")
def ai_search(payload: SearchQuery, db: Session = Depends(get_db)):
    """Natural-language search across all farmers' listings, with an
    LLM-generated comparison summary (RAG: retrieval from the Product table
    + augmented generation)."""
    return run_ai_search(db, payload.query, payload.max_results)


# ---- Orders --------------------------------------------------------------
orders_router = APIRouter(prefix="/api/v1/orders", tags=["orders"])


@orders_router.post("/", response_model=OrderOut)
def place_order(payload: OrderCreate, db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    if current_user.role != UserRole.CUSTOMER:
        raise HTTPException(status_code=403, detail="Only customers can place orders")

    try:
        order = create_order(
            db, current_user.id, payload.farmer_id,
            payload.fulfillment_type, payload.items,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return order


@orders_router.get("/{order_id}", response_model=OrderOut)
def get_order(order_id: str, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


# ---- Payments (incl. return/refund policy enforcement) -------------------
payments_router = APIRouter(prefix="/api/v1/payments", tags=["payments"])


@payments_router.post("/", response_model=PaymentOut)
def capture_payment(payload: PaymentCreate, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == payload.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    payment = Payment(
        order_id=order.id,
        amount=order.total_amount,
        method=payload.method,
        status=PaymentStatus.PAID,  # in production: set after gateway webhook confirms
    )
    db.add(payment)
    db.commit()
    db.refresh(payment)
    return payment


@payments_router.post("/refund-request")
def request_refund(payload: RefundRequestCreate, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == payload.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    try:
        req = file_refund_request(db, order, payload.reason, payload.evidence_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"refund_request_id": req.id, "status": "pending_review"}


@payments_router.post("/refund-request/{request_id}/approve")
def admin_approve_refund(request_id: str, full: bool = True, db: Session = Depends(get_db)):
    req = db.query(RefundRequest).filter(RefundRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Refund request not found")

    try:
        payment = approve_refund(db, req, full=full)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "payment_id": payment.id,
        "status": payment.status,
        "refunded_amount": payment.refunded_amount,
    }


# ---- Reviews ---------------------------------------------------------------
reviews_router = APIRouter(prefix="/api/v1/reviews", tags=["reviews"])


@reviews_router.post("/")
def create_review(payload: ReviewCreate, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    order = db.query(Order).filter(Order.id == payload.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if order.status != OrderStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Can only review completed orders")

    review = Review(
        order_id=order.id,
        reviewer_id=current_user.id,
        farmer_id=order.farmer_id,
        rating=payload.rating,
        comment=payload.comment,
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return {"id": review.id, "rating": review.rating}


# ---- Favorites (optional, per SRS) ---------------------------------------
favorites_router = APIRouter(prefix="/api/v1/favorites", tags=["favorites"])


@favorites_router.post("/")
def add_favorite(payload: FavoriteCreate, db: Session = Depends(get_db),
                  current_user: User = Depends(get_current_user)):
    fav = Favorite(user_id=current_user.id, product_id=payload.product_id)
    db.add(fav)
    db.commit()
    db.refresh(fav)
    return {"id": fav.id}


@favorites_router.get("/")
def list_favorites(db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    return db.query(Favorite).filter(Favorite.user_id == current_user.id).all()


# ======================================================================
# 10. APP ENTRYPOINT
# ======================================================================

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AgriConnect AI API", version="1.0.0")

app.include_router(users_router)
app.include_router(products_router)
app.include_router(search_router)
app.include_router(orders_router)
app.include_router(payments_router)
app.include_router(reviews_router)
app.include_router(favorites_router)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "AgriConnect AI API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agriconnect_ai_app:app", host="0.0.0.0", port=8000, reload=True)
