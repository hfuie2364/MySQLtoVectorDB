# MySQL to Vector DB

Spring Boot application that reads QA data from MySQL, creates embeddings through an external embedding API, and stores the result in a Milvus collection.

## Features

- `GET /qa`: Query source QA rows from MySQL.
- `GET /milvus-data`: Query rows stored in the configured Milvus collection.
- `GET /ingest`: Delete existing Milvus collection data, then re-ingest QA rows from MySQL.

## Project Structure

```text
mysql-to-milvus/
  src/main/java/com/hanwha/mysqltomilvus/
    MysqlToMilvusApplication.java
    QaController.java
  src/main/resources/application.yml
  pom.xml
```

## Requirements

- Java 17
- Milvus running and reachable
- MySQL running and reachable
- Embedding API endpoint and API key

## Configuration

Runtime configuration is managed in `mysql-to-milvus/src/main/resources/application.yml`.

Sensitive values are read from environment variables:

```powershell
$env:MYSQL_URL="jdbc:mysql://HOST:PORT/DB_NAME?useSSL=false&serverTimezone=Asia/Seoul&characterEncoding=UTF-8"
$env:MYSQL_USERNAME="your_mysql_username"
$env:MYSQL_PASSWORD="your_mysql_password"
$env:EMBEDDING_ENDPOINT="https://your-embedding-api/embed"
$env:EMBEDDING_API_KEY="your_embedding_api_key"
$env:MILVUS_HOST="localhost"
$env:MILVUS_PORT="19530"
$env:MILVUS_COLLECTION_NAME="moon_test"
```

The MySQL table/column mapping, Milvus field mapping, query limits, max string lengths, and ingest batch size are also configurable in `application.yml`.

## Run

```powershell
cd mysql-to-milvus
.\mvnw.cmd spring-boot:run
```

Default server port:

```text
http://localhost:8080
```

## API

```text
GET /qa
GET /milvus-data
GET /ingest
```

## Notes

- `/ingest` clears the configured Milvus collection data before inserting new rows.
- The Milvus collection must already exist with fields matching the configured names.
- Do not commit real database passwords or API keys.
