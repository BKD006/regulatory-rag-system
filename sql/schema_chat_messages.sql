CREATE TABLE public.qa_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_qa_logs_time
ON public.qa_logs(created_at);