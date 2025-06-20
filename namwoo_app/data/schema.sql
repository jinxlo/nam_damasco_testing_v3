-- Namwoo Database Schema (Damasco Version) - REVISED and COMPLETE
-- PostgreSQL dialect

-- 1. Enable pgvector extension (Required for vector similarity search)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Table for storing product information synced from Damasco API
-- This table will now store each product instance at a specific warehouse.
DROP TABLE IF EXISTS products CASCADE; -- Drop existing table if it exists to apply new PK and constraints
CREATE TABLE products (
    -- Composite Primary Key: A unique identifier for a product at a specific warehouse.
    -- We'll create this in the application (e.g., "D0007277_Almacen_San_Martin_1")
    -- and store it here.
    id VARCHAR(512) PRIMARY KEY, -- Increased length for "itemcode_warehousename"

    item_code VARCHAR(64) NOT NULL,      -- Original Item Code from Damasco (e.g., D0007277)
    item_name TEXT NOT NULL,             -- Product name
    
    -- Descriptive attributes for the product itself
    description TEXT,                    -- Raw HTML product description from Damasco
    specifitacion TEXT,                  -- Detailed product specifications
    category VARCHAR(128),               -- Main category
    sub_category VARCHAR(128),           -- Sub-category
    brand VARCHAR(128),                  -- Brand of the product
    line VARCHAR(128),                   -- Product line (from Damasco)
    item_group_name VARCHAR(128),        -- Broader group name (from Damasco)

    -- Location-specific attributes for this entry
    warehouse_name VARCHAR(255) NOT NULL, -- Warehouse name (from Damasco's 'whsName')
    branch_name VARCHAR(255),            -- Branch name (from Damasco's 'branchName')
    
    -- Financial and stock attributes
    price NUMERIC(12, 2),                -- Price, typically in primary currency (e.g., USD)
    price_bolivar NUMERIC(12, 2),        -- Price in Bolívares (Bs.)
    stock INTEGER DEFAULT 0,             -- Current stock quantity at this specific warehouse

    -- Embedding related fields
    searchable_text_content TEXT,        -- The actual text string used to generate the embedding
    embedding vector(1536),              -- Vector embedding (dimension from config.EMBEDDING_DIMENSION)
    
    -- Auditing and additional data
    source_data_json JSONB DEFAULT '{}'::jsonb, -- Original JSON for this specific product-warehouse entry
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, -- Creation timestamp with timezone
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP  -- Last update timestamp with timezone
);

-- Make the combination of item_code and warehouse_name unique
ALTER TABLE products
ADD CONSTRAINT uq_item_code_per_warehouse UNIQUE (item_code, warehouse_name);

-- 3. Indexes for common filters
CREATE INDEX IF NOT EXISTS idx_products_item_code ON products (item_code); -- For finding all locations of an item_code
CREATE INDEX IF NOT EXISTS idx_products_brand ON products (brand);
CREATE INDEX IF NOT EXISTS idx_products_category ON products (category);
CREATE INDEX IF NOT EXISTS idx_products_warehouse_name ON products (warehouse_name);
CREATE INDEX IF NOT EXISTS idx_products_branch_name ON products (branch_name);
-- No need for item_name index if primarily using vector search for names/descriptions

-- 4. Vector Similarity Index (HNSW for pgvector)
-- Ensure this matches your typical query needs (e.g., vector_cosine_ops or vector_l2_ops)
CREATE INDEX IF NOT EXISTS idx_products_embedding_hnsw ON products USING hnsw (embedding vector_cosine_ops);

-- 5. Table for storing human takeover pause state per Support Board conversation (Unchanged)
CREATE TABLE IF NOT EXISTS conversation_pauses (
    conversation_id VARCHAR(255) PRIMARY KEY,
    paused_until TIMESTAMP WITH TIME ZONE NOT NULL
);

-- 6. Optional Comments
COMMENT ON TABLE products IS 'Stores specific product stock entries at each warehouse, synchronized from Damasco API, including vector embeddings for semantic search of the product description.';
COMMENT ON COLUMN products.id IS 'Application-generated composite PK: item_code + sanitized warehouse_name.';
COMMENT ON COLUMN products.embedding IS 'Vector embedding generated from product descriptive text (brand, name, category, etc.).';
COMMENT ON COLUMN products.warehouse_name IS 'The specific warehouse where this stock entry is located (from Damasco whsName).';

-- 7. Permissions (Uncomment and adjust username if needed)
-- DO THIS MANUALLY IN YOUR DB or ensure your Docker setup handles permissions for 'namwoo' user
-- GRANT ALL PRIVILEGES ON TABLE products TO namwoo;
-- GRANT ALL PRIVILEGES ON TABLE conversation_pauses TO namwoo;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO namwoo; -- If any sequences are used (not with string PKs like this)


-- Function to automatically update 'updated_at' timestamp (Optional but good practice)
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_products_timestamp
BEFORE UPDATE ON products
FOR EACH ROW
EXECUTE FUNCTION trigger_set_timestamp();

-- End of schema definition