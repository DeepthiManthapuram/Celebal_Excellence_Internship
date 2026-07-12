import os

print("=" * 50)
print("RAG SYSTEM VALIDATION")
print("=" * 50)

# Check data folder
if os.path.exists("data"):
    print("PASS: Data folder found")
else:
    print("FAIL: Data folder missing")

# Check vectorstore folder
if os.path.exists("vectorstore"):
    print("PASS: Vectorstore folder found")
else:
    print("FAIL: Vectorstore folder missing")

# Check FAISS index
if os.path.exists("vectorstore/index.faiss"):
    print("PASS: FAISS index created")
else:
    print("FAIL: FAISS index missing")

# Check metadata file
if os.path.exists("vectorstore/index.pkl"):
    print("PASS: FAISS metadata file created")
else:
    print("FAIL: FAISS metadata missing")

# Check log file
if os.path.exists("rag.log"):
    print("PASS: Log file created")
else:
    print("FAIL: Log file missing")

# Check metrics report
if os.path.exists("metrics_report.txt"):
    print("PASS: Metrics report created")
else:
    print("FAIL: Metrics report missing")

print("\nValidation Completed Successfully!")