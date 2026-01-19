CREATE TABLE IF NOT EXISTS company_profile (
  id SERIAL PRIMARY KEY,
  name TEXT,
  description TEXT,
  country TEXT,
  website TEXT,
  notes TEXT,
  source_notes TEXT
);

CREATE TABLE IF NOT EXISTS company_capability (
  id SERIAL PRIMARY KEY,
  profile_id INT REFERENCES company_profile(id),
  capability TEXT NOT NULL,
  capacity TEXT,
  certification TEXT,
  source_notes TEXT
);

CREATE TABLE IF NOT EXISTS tenders (
  id SERIAL PRIMARY KEY,
  tender_id TEXT UNIQUE NOT NULL,
  source_file TEXT NOT NULL,
  primary_title TEXT,
  summary TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS company_delivery (
  id SERIAL PRIMARY KEY,
  tender_id TEXT REFERENCES tenders(tender_id),
  title TEXT NOT NULL,
  customer TEXT,
  delivered_at DATE,
  value NUMERIC,
  currency TEXT,
  scope TEXT,
  evidence TEXT,
  source_notes TEXT
);

CREATE TABLE IF NOT EXISTS tender_lots (
  id SERIAL PRIMARY KEY,
  tender_id TEXT REFERENCES tenders(tender_id),
  lot_number TEXT,
  title TEXT,
  description TEXT,
  estimated_value NUMERIC,
  estimated_currency TEXT,
  qualification_min_value NUMERIC,
  qualification_currency TEXT,
  pages INT[],
  source_file TEXT
);

CREATE TABLE IF NOT EXISTS tender_activity (
  id SERIAL PRIMARY KEY,
  tender_id TEXT REFERENCES tenders(tender_id),
  status TEXT,
  notes TEXT,
  source_notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_tender_lots_tender_id ON tender_lots(tender_id);
CREATE INDEX IF NOT EXISTS idx_tender_lots_title ON tender_lots USING gin (to_tsvector('simple', coalesce(title,'')));
CREATE INDEX IF NOT EXISTS idx_tender_lots_description ON tender_lots USING gin (to_tsvector('simple', coalesce(description,'')));
CREATE INDEX IF NOT EXISTS idx_company_delivery_tender_id ON company_delivery(tender_id);
CREATE INDEX IF NOT EXISTS idx_company_delivery_text ON company_delivery USING gin (
  to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(scope,'') || ' ' || coalesce(customer,''))
);
