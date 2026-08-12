import { createClient } from '@supabase/supabase-js'

const SUPABASE_URL  = 'https://wzjtyosijxacgytfmjtu.supabase.co'
const SUPABASE_ANON = 'sb_publishable_pI6VsKC_bF3PwGDRpQ9U5w_TmoL7uB3'

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON)