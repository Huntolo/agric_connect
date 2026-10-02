# AgriConnect

A farmer-to-customer marketplace with natural-language search.

- **Backend:** `agriconnect.py` (FastAPI + SQLAlchemy + SQLite)
- **Front end:** `index.html` (one file, no build step)
- **AI search:** an LLM reads a request like *"cheap organic tomatoes under 2000"*, the app retrieves matching live listings from farmers, ranks them by price and rating, and writes a short recommendation. Without an OpenAI key it falls back to simple rule-based matching.

---

## 1. What you need

- **Python 3.10 or newer.** Get it from https://www.python.org/downloads/ and **tick "Add python.exe to PATH"** during install.
- **VS Code** with the **Python** extension.
- An **OpenAI API key** (optional; only for the real AI behavior).

Check Python works:

```powershell
python --version
```

If that fails on Windows, try `py --version` and use `py` in place of `python` below.

---

## 2. Project folder

Put both files in one folder:

```
agriconnect/
├── agriconnect.py
├── index.html
├── requirements.txt     (you create this in step 3)
└── .env                 (optional, step 5)
```

In VS Code: **File → Open Folder** and choose this folder. Open a terminal with `` Ctrl+` ``. The terminal path should end in `agriconnect`.

---

## 3. Install

Create a virtual environment and activate it:

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
If PowerShell blocks the script, run this once and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` at the start of the terminal line.

Create a file named `requirements.txt` with this content:

```
fastapi
uvicorn[standard]
sqlalchemy
pydantic
pydantic-settings
python-jose[cryptography]
passlib[bcrypt]
bcrypt==4.0.1
python-multipart
email-validator
openai
```

Then install:

```powershell
python -m pip install -r requirements.txt
```

`bcrypt==4.0.1` and `email-validator` are there on purpose. Newer bcrypt versions break password hashing in passlib, and `email-validator` is needed for the email field.

---

## 4. Add the front end routes to the backend (one time)

The original `agriconnect.py` has no endpoint to create a farmer profile, so farmers cannot add products. This patch adds that, plus three small routes the front end uses.

Open `agriconnect.py`. Find these lines near the bottom:

```python
app.include_router(favorites_router)


@app.get("/")
def health_check():
```

Paste this block **between** them:

```python
# ---- Front end support ----
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class FarmerProfileCreate(BaseModel):
    farm_name: str
    certifications: Optional[str] = None
    delivery_radius_km: float = 10.0


@app.get("/api/v1/users/me")
def read_me(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    profile = db.query(FarmerProfile).filter(FarmerProfile.user_id == current_user.id).first()
    return {
        "id": current_user.id,
        "name": current_user.name,
        "role": current_user.role.value,
        "has_profile": profile is not None,
        "farmer_profile_id": profile.id if profile else None,
        "farm_name": profile.farm_name if profile else None,
    }


@app.post("/api/v1/farmers/profile")
def create_farmer_profile(payload: FarmerProfileCreate, db: Session = Depends(get_db),
                          current_user: User = Depends(get_current_user)):
    if current_user.role != UserRole.FARMER:
        raise HTTPException(status_code=403, detail="Only farmers can create a farm profile")
    if db.query(FarmerProfile).filter(FarmerProfile.user_id == current_user.id).first():
        raise HTTPException(status_code=400, detail="Profile already exists")
    profile = FarmerProfile(user_id=current_user.id, farm_name=payload.farm_name,
                            certifications=payload.certifications,
                            delivery_radius_km=payload.delivery_radius_km)
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return {"id": profile.id, "farm_name": profile.farm_name}


@app.get("/api/v1/marketplace")
def marketplace(db: Session = Depends(get_db)):
    return rank_matches(retrieve_matching_products(
        db, keywords="", max_price=None, organic_only=False, category=None, limit=100))


@app.get("/app")
def frontend():
    return FileResponse(Path(__file__).parent / "index.html")
```

Also fix the last line of the file. Change `agriconnect_ai_app:app` to `agriconnect:app`:

```python
uvicorn.run("agriconnect:app", host="0.0.0.0", port=8000, reload=True)
```

Save the file.

---

## 5. Optional: turn on real AI search

Create a file named `.env` in the project folder:

```
OPENAI_API_KEY=sk-your-key-here
```

Never share this file or commit it to GitHub. Without it, search still works using simple keyword matching and a templated summary.

---

## 6. Run it

From the project folder, with `(venv)` active:

```powershell
python -m uvicorn agriconnect:app --reload
```

Wait for `Application startup complete`. Leave this terminal open while you use the app. Press `Ctrl+C` to stop it.

Then open:

| Address | What it is |
|---|---|
| http://localhost:8000/app | **The AgriConnect website** |
| http://localhost:8000/docs | Interactive API docs (for testing endpoints) |
| http://localhost:8000/ | Health check (returns `{"status": "ok"}`) |

To start it again later: open the folder in VS Code, open a terminal, run `venv\Scripts\Activate.ps1`, then the `uvicorn` command above.

---

## 7. Try it out

1. Open http://localhost:8000/app and click **Sign up**.
2. Choose **Sell produce**, fill in your details (email and phone are both required) and create the account.
3. Go to **Sell your produce**, enter your farm name, then add a few listings, for example:
   - Fresh tomatoes, Vegetables, kg, ₦1800, stock 50, organic
   - Pepper, Vegetables, kg, ₦1200, stock 30
4. Click **Log out**, then sign up again as **Buy produce** (use a different email and phone).
5. On the **Market** page, type `cheap organic tomatoes under 2000` and click **Find produce**.
6. Add something to the cart, open **Cart**, choose pickup or delivery, and click **Place order**.
7. Open **My orders** to see the order and its status.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not import module "agriconnect"` | Your terminal is in the wrong folder. Run `dir` and make sure you can see `agriconnect.py`, then `cd` into that folder. Also check the file isn't named `agriconnect.py.txt`. |
| `python` is not recognized / `-m` command not found | Python isn't installed or isn't on PATH. Reinstall it with "Add python.exe to PATH" ticked, then fully restart VS Code. Or use `py` instead of `python`. |
| Activation script blocked in PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| `No such option '--reload/c:/...'` | The command got mangled. Clear the line with `Ctrl+C` and type the command by hand instead of pasting. |
| `No module named ...` | The packages went into a different environment. Make sure `(venv)` shows, then run `python -m pip install -r requirements.txt`. |
| Error mentioning `email-validator` | `python -m pip install email-validator` |
| Errors about bcrypt or "password cannot be longer than 72 bytes" at signup | `python -m pip install bcrypt==4.0.1` |
| Page shows a yellow banner: "missing the front end routes" | Step 4 wasn't saved correctly. Re-paste the patch, save, and refresh. |
| Page shows "Can't reach the server" | The server isn't running. Start it (step 6). |
| `Address already in use` / port 8000 busy | Close the other terminal running uvicorn, or use another port: `python -m uvicorn agriconnect:app --reload --port 8001` and open `localhost:8001/app`. |
| `favicon.ico 404` in the terminal | Harmless. Browsers ask for a tab icon and the app has none. |
| "User already exists" at signup | That email or phone is already registered. Use different ones. |
| Search says no results | Nothing matches. Add listings as a farmer first, or try fewer words and a higher budget. |

**Start fresh:** stop the server and delete `agriconnect.db` in the project folder. A new empty database is created on the next start.

---

## 9. Known limitations

- **My orders** only lists orders placed from the same browser, because the backend has no "list my orders" endpoint.
- **Reviews** can't be left yet, because there is no endpoint to mark an order as completed. "Report a problem" works.
- **Payments** are simulated. Choosing a payment method marks the order as paid immediately; no real gateway is connected.
- **Delivery fee** is a flat `3.0` per order in the backend (`create_order`). Change it there if you want a different amount.
- Before going live, set a strong `JWT_SECRET` in `.env` (the default is a placeholder) and replace `allow_origins=["*"]` with your real site address.

---

## 10. Project reference

| Feature | Endpoint |
|---|---|
| Register / log in | `POST /api/v1/users/register`, `POST /api/v1/users/login` |
| Current user | `GET /api/v1/users/me` |
| Farm profile | `POST /api/v1/farmers/profile` |
| Products | `POST /api/v1/products/`, `GET /api/v1/products/` |
| All listings with farm names | `GET /api/v1/marketplace` |
| AI search | `POST /api/v1/search/` |
| Orders | `POST /api/v1/orders/`, `GET /api/v1/orders/{id}` |
| Payments and refunds | `POST /api/v1/payments/`, `POST /api/v1/payments/refund-request` |
| Reviews | `POST /api/v1/reviews/` |
