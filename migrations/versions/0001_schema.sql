CREATE TABLE documents (
	id VARCHAR NOT NULL, 
	tenant_id VARCHAR, 
	title VARCHAR, 
	source_path VARCHAR, 
	version VARCHAR, 
	status VARCHAR, 
	effective_at VARCHAR, 
	expires_at VARCHAR, 
	visibility VARCHAR, 
	PRIMARY KEY (id)
);
CREATE TABLE evaluation_runs (
	id VARCHAR NOT NULL, 
	tenant_id VARCHAR, 
	user_id VARCHAR, 
	payload TEXT, 
	PRIMARY KEY (id)
);
CREATE TABLE tenants (
	id VARCHAR NOT NULL, 
	name VARCHAR, 
	created_at VARCHAR, 
	PRIMARY KEY (id)
);
CREATE TABLE traces (
	id VARCHAR NOT NULL, 
	tenant_id VARCHAR, 
	user_id VARCHAR, 
	conversation_id VARCHAR, 
	payload TEXT, 
	PRIMARY KEY (id)
);
CREATE TABLE document_chunks (
	chunk_id VARCHAR NOT NULL, 
	document_id VARCHAR, 
	chunk_index INTEGER, 
	content TEXT, 
	metadata_json VARCHAR, 
	content_hash VARCHAR, 
	PRIMARY KEY (chunk_id), 
	FOREIGN KEY(document_id) REFERENCES documents (id)
);
CREATE TABLE products (
	id VARCHAR NOT NULL, 
	tenant_id VARCHAR, 
	name VARCHAR, 
	category VARCHAR, 
	price_cents INTEGER, 
	stock INTEGER, 
	attributes_json VARCHAR, 
	PRIMARY KEY (id), 
	FOREIGN KEY(tenant_id) REFERENCES tenants (id)
);
CREATE TABLE users (
	id VARCHAR NOT NULL, 
	tenant_id VARCHAR, 
	display_name VARCHAR, 
	role VARCHAR, 
	PRIMARY KEY (id), 
	FOREIGN KEY(tenant_id) REFERENCES tenants (id)
);
CREATE TABLE auth_sessions (
	token_hash VARCHAR NOT NULL, 
	user_id VARCHAR, 
	expires_at VARCHAR, 
	revoked_at VARCHAR, 
	PRIMARY KEY (token_hash), 
	FOREIGN KEY(user_id) REFERENCES users (id)
);
CREATE TABLE conversations (
	id VARCHAR NOT NULL, 
	tenant_id VARCHAR, 
	user_id VARCHAR, 
	title VARCHAR, 
	created_at VARCHAR, 
	updated_at VARCHAR, 
	PRIMARY KEY (id), 
	FOREIGN KEY(tenant_id) REFERENCES tenants (id), 
	FOREIGN KEY(user_id) REFERENCES users (id)
);
CREATE TABLE orders (
	id VARCHAR NOT NULL, 
	tenant_id VARCHAR, 
	user_id VARCHAR, 
	status VARCHAR, 
	paid_amount_cents INTEGER, 
	paid_at VARCHAR, 
	delivered_at VARCHAR, 
	version INTEGER, 
	PRIMARY KEY (id), 
	FOREIGN KEY(tenant_id) REFERENCES tenants (id), 
	FOREIGN KEY(user_id) REFERENCES users (id)
);
CREATE TABLE messages (
	id VARCHAR NOT NULL, 
	conversation_id VARCHAR, 
	role VARCHAR, 
	content TEXT, 
	created_at VARCHAR, 
	trace_id VARCHAR, 
	PRIMARY KEY (id), 
	FOREIGN KEY(conversation_id) REFERENCES conversations (id)
);
CREATE TABLE order_items (
	id VARCHAR NOT NULL, 
	order_id VARCHAR, 
	product_id VARCHAR, 
	quantity INTEGER, 
	unit_price_cents INTEGER, 
	PRIMARY KEY (id), 
	FOREIGN KEY(order_id) REFERENCES orders (id), 
	FOREIGN KEY(product_id) REFERENCES products (id)
);
CREATE TABLE refund_requests (
	id VARCHAR NOT NULL, 
	tenant_id VARCHAR, 
	user_id VARCHAR, 
	order_id VARCHAR, 
	amount_cents INTEGER, 
	status VARCHAR, 
	idempotency_key VARCHAR, 
	policy_version VARCHAR, 
	created_at VARCHAR, 
	PRIMARY KEY (id), 
	FOREIGN KEY(tenant_id) REFERENCES tenants (id), 
	FOREIGN KEY(user_id) REFERENCES users (id), 
	UNIQUE (order_id), 
	FOREIGN KEY(order_id) REFERENCES orders (id), 
	UNIQUE (idempotency_key)
);
CREATE TABLE shipments (
	id VARCHAR NOT NULL, 
	order_id VARCHAR, 
	carrier VARCHAR, 
	tracking_no VARCHAR, 
	status VARCHAR, 
	updated_at VARCHAR, 
	PRIMARY KEY (id), 
	FOREIGN KEY(order_id) REFERENCES orders (id)
);
CREATE TABLE tickets (
	id VARCHAR NOT NULL, 
	tenant_id VARCHAR, 
	user_id VARCHAR, 
	conversation_id VARCHAR, 
	summary VARCHAR, 
	reason VARCHAR, 
	status VARCHAR, 
	created_at VARCHAR, 
	PRIMARY KEY (id), 
	FOREIGN KEY(tenant_id) REFERENCES tenants (id), 
	FOREIGN KEY(user_id) REFERENCES users (id), 
	FOREIGN KEY(conversation_id) REFERENCES conversations (id)
);
CREATE TABLE workflow_instances (
	id VARCHAR NOT NULL, 
	tenant_id VARCHAR, 
	user_id VARCHAR, 
	conversation_id VARCHAR, 
	graph_thread_id VARCHAR, 
	type VARCHAR, 
	status VARCHAR, 
	order_id VARCHAR, 
	amount_cents INTEGER, 
	policy_version VARCHAR, 
	confirm_nonce_hash VARCHAR, 
	nonce_expires_at VARCHAR, 
	operation_key VARCHAR, 
	order_version INTEGER, 
	version INTEGER, 
	updated_at VARCHAR, 
	PRIMARY KEY (id), 
	FOREIGN KEY(tenant_id) REFERENCES tenants (id), 
	FOREIGN KEY(user_id) REFERENCES users (id), 
	FOREIGN KEY(conversation_id) REFERENCES conversations (id), 
	FOREIGN KEY(order_id) REFERENCES orders (id), 
	UNIQUE (operation_key)
);
