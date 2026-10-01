AgriConnect  — Farmer-to-Customer Marketplace with Powered Search
(FastAPI + SQLAlchemy + LLM query interpretation + RAG comparison pipeline)

This merges two things into one app:
  1. The original AgriConnect farmer-to-customer marketplace (auth, farmer
     profiles, product listings, cart/orders, payments, reviews, and the
     return/refund policy workflow).2.
     The coursework requirements, re-scoped to this domain: instead of
     comparing prices across external retailers (Jumia/Konga/Amazon), the
     platform uses an LLM to interpret natural-language customer queries and
     a RAG pipeline that retrieves REAL, live listings from AgriConnect's own
     farmers, compares them, and generates an  summary recommending the
     best match(es).

     Example: "cheap organic tomatoes near me under 2000"
       -> LLM interprets: keywords="tomatoes", organic=True, max_price=2000
       -> RAG retrieval: query the Product table across all active farmers
       -> Comparison engine: rank by price + farmer rating
       -> LLM generation: "Farmer Ade's organic tomatoes at ₦1,800/kg offer
          the best value and have a 4.8 rating, though Farmer Musa's stock
          is slightly cheaper per kg if freshness matters less...
