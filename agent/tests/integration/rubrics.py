BREAKDOWN_RUBRIC = """
Pass if the response is a structured breakdown that covers:
- buyer/procedure
- scope
- lots/deliverables
- timeline/deadlines
- qualification requirements/eligibility
- securities/payment terms
- submission details
If a section is not present in the tender, the response should say it is not found or not covered.
Fail if it is a generic summary without sections or misses most required sections.
""".strip()

READINESS_RUBRIC = """
Pass if the response provides:
- a readiness verdict (high/medium/low OR likely/uncertain/unlikely/unknown)
- key gaps/risks or missing evidence
- evidence or citations from the tender (file#page) and/or company_db
Fail if there is no verdict or no evidence-based gaps.
""".strip()

SIMILAR_RUBRIC = """
Pass if the response lists 3-5 similar tenders and for each provides:
- why it is similar (scope, buyer, CPV, geography, etc.)
- evidence via snippets or citations when possible
Fail if fewer than 3 tenders are provided or reasons are missing.
""".strip()
