-- 1. Tabellen erstellen
CREATE TABLE products (
  product_id SERIAL PRIMARY KEY,
  name         VARCHAR(100) NOT NULL,
  category     VARCHAR(50),
  unit_price   NUMERIC(10,2) NOT NULL
);

CREATE TABLE sales (
  sale_id     SERIAL PRIMARY KEY,
  product_id  INT NOT NULL REFERENCES products(product_id),
  quantity    INT NOT NULL,
  sale_date   DATE NOT NULL,
  total_price NUMERIC(12,2) NOT NULL
);

-- 2. Beispiel‑Produkte einfügen
INSERT INTO products (name, category, unit_price) VALUES
  ('Kaffeemaschine A',    'Haushalt',  79.99),
  ('Espressobohnen 1kg',   'Lebensmittel', 14.50),
  ('Teekanne Edelstahl',   'Haushalt',  29.90),
  ('Kaffee Tasse Set (4)', 'Haushalt',  19.99),
  ('Filterpapier 100 St.',  'Zubehör',    3.49),
  ('French Press 1L',      'Haushalt',  24.99),
  ('Latte Macchiato Glas', 'Gläser',    12.50),
  ('Milchaufschäumer',     'Geräte',    34.90),
  ('Vanille Sirup 500ml',  'Zubehör',    6.99),
  ('Schokoladensirup 500ml','Zubehör',   6.99);

-- 3. Beispiel‑Verkäufe einfügen (letzte 30 Tage: 2025-03-19 bis 2025-04-18)
INSERT INTO sales (product_id, quantity, sale_date, total_price) VALUES
  (1,  2, '2025-04-01', 2 * 79.99),
  (2,  5, '2025-04-02', 5 * 14.50),
  (3,  1, '2025-04-02', 1 * 29.90),
  (4, 10, '2025-04-03', 10 * 19.99),
  (5, 20, '2025-04-04', 20 * 3.49),
  (6,  3, '2025-04-05', 3 * 24.99),
  (7,  4, '2025-04-06', 4 * 12.50),
  (8,  2, '2025-04-07', 2 * 34.90),
  (9,  6, '2025-04-08', 6 * 6.99),
  (10, 7, '2025-04-09', 7 * 6.99),
  (1,  1, '2025-03-25', 1 * 79.99),
  (2,  3, '2025-03-26', 3 * 14.50),
  (3,  2, '2025-03-27', 2 * 29.90),
  (4,  5, '2025-03-28', 5 * 19.99),
  (5, 10, '2025-03-29', 10 * 3.49),
  (6,  2, '2025-03-30', 2 * 24.99),
  (7,  3, '2025-03-31', 3 * 12.50),
  (8,  1, '2025-04-10', 1 * 34.90),
  (9,  2, '2025-04-11', 2 * 6.99),
  (10, 3, '2025-04-12', 3 * 6.99),
  (1,  4, '2025-04-13', 4 * 79.99),
  (2,  6, '2025-04-14', 6 * 14.50),
  (3,  1, '2025-04-15', 1 * 29.90),
  (4,  2, '2025-04-16', 2 * 19.99),
  (5, 15, '2025-04-17', 15 * 3.49),
  (6,  1, '2025-04-18', 1 * 24.99);
