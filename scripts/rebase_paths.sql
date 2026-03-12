-- Rebase file paths from one prefix to another.
-- Usage: set the two variables below, then run against the DB.
--
--   psql -v old='/Users/tulas/Projects/stripes-rag/docs' -v new='/docs' -f scripts/rebase_paths.sql

\set old_prefix :'old'
\set new_prefix :'new'

BEGIN;

UPDATE file_tracking
SET file_path = :'new_prefix' || SUBSTRING(file_path FROM LENGTH(:'old_prefix') + 1)
WHERE file_path LIKE :'old_prefix' || '%';

UPDATE document_chunks
SET source_file = :'new_prefix' || SUBSTRING(source_file FROM LENGTH(:'old_prefix') + 1)
WHERE source_file LIKE :'old_prefix' || '%';

COMMIT;
