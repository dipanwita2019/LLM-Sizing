import hashlib
password = "EdgeAI123"  # Your chosen password
hashed = hashlib.sha256(password.encode()).hexdigest()
print(hashed)