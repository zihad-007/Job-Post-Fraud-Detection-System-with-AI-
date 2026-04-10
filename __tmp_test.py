from app import app
c = app.test_client()
with c.session_transaction() as s:
    s['user_id']=1
    s['username']='testuser'
resp = c.get('/dashboard')
print(resp.status_code)
print(resp.data[:400])
