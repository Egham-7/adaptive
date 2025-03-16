export type APIKey = {
  id: string;
  name: string;
  user_id: string;
  key_prefix: string;
  status: string;
  created_at: string;
  last_used_at: string;
  expires_at?: string;
};

export type CreateAPIKeyRequest = {
  name: string;
  user_id: string;
  status: string;
  expires_at?: string;
};

export type UpdateAPIKeyRequest = {
  name: string;
  status: string;
};

export type CreateAPIKeyResponse = {
  api_key: APIKey;
  full_api_key: string;
};
