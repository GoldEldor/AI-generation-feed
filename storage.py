def get_history(session_id: str, limit: int = 12):
    con = _db()
    rows = con.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    con.close()
    
    # If local storage is empty (e.g. after a Colab/server restart), pull from Supabase cloud!
    if not rows and SUPABASE_URL and SUPABASE_KEY:
        try:
            import requests
            resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?session_id=eq.{session_id}&order=created_at.desc&limit={limit}",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                },
                timeout=5
            )
            if resp.status_code == 200:
                cloud_data = resp.json()
                # Supabase returns newest first; reverse it back to chronological order
                return [{"role": r["role"], "content": r["content"]} for r in reversed(cloud_data)]
        except Exception as exc:
            print(f"[storage] Failed to fetch history from Supabase: {exc}")

    return [{"role": r, "content": c} for r, c in reversed(rows)]
  
