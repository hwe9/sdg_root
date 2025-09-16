# /sdg_root/src/auth/token_store.py
import os, time, json
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
_r = redis.from_url(REDIS_URL, decode_responses=True)

FAMILY_KEY = "refresh:family:{sub}"
TOKEN_KEY  = "refresh:token:{jti}"

def register_refresh_token(sub: str, jti: str, exp_ts: int):
    _r.sadd(FAMILY_KEY.format(sub=sub), jti)
    ttl = max(1, exp_ts - int(time.time()))
    _r.hset(TOKEN_KEY.format(jti=jti), mapping={"status": "active", "sub": sub})
    _r.expire(TOKEN_KEY.format(jti=jti), ttl)
    _r.expire(FAMILY_KEY.format(sub=sub), ttl)

def consume_refresh_token(jti: str) -> str:
    data = _r.hgetall(TOKEN_KEY.format(jti=jti))
    if not data:
        return "unknown"
    status = data.get("status", "unknown")
    if status == "used":
        # mark family as compromised
        sub = data.get("sub")
        if sub:
            family = _r.smembers(FAMILY_KEY.format(sub=sub))
            for t in family:
                _r.hset(TOKEN_KEY.format(jti=t), mapping={"status": "revoked"})
        return "reused"
    if status == "revoked":
        return "revoked"
    _r.hset(TOKEN_KEY.format(jti=jti), mapping={"status": "used"})
    return "ok"
