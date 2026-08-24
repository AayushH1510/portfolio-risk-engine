-- Run this once in the Supabase SQL editor to create the feedback table
-- used by src/components/FeedbackButton.jsx.

create table if not exists feedback (
  id         uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  user_id    uuid references auth.users(id) on delete set null,
  type       text not null default 'Other' check (type in ('Bug', 'Confusing', 'Suggestion', 'Other')),
  message    text not null,
  page       text
);

-- Row-level security: allow anyone (signed in or not) to submit feedback,
-- but don't expose read access through the client — feedback is meant to
-- be reviewed from the Supabase dashboard, not the app.
alter table feedback enable row level security;

create policy "Anyone can submit feedback"
  on feedback for insert
  to anon, authenticated
  with check (true);
