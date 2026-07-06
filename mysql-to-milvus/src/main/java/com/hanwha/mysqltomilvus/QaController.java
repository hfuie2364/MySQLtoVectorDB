package com.hanwha.mysqltomilvus;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.DescribeCollectionResponse;
import io.milvus.grpc.FieldSchema;
import io.milvus.grpc.QueryResults;
import io.milvus.param.ConnectParam;
import io.milvus.param.R;
import io.milvus.param.collection.DescribeCollectionParam;
import io.milvus.param.collection.FlushParam;
import io.milvus.param.collection.HasCollectionParam;
import io.milvus.param.collection.LoadCollectionParam;
import io.milvus.param.dml.DeleteParam;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.QueryParam;
import io.milvus.response.QueryResultsWrapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@RestController
public class QaController {

    private final JdbcTemplate jdbcTemplate;
    private final MilvusServiceClient milvusClient;
    private final String collectionName;
    private final String embeddingEndpoint;
    private final String embeddingApiKey;
    private final String qaTableName;
    private final String qaIdColumn;
    private final String companyIdColumn;
    private final String questionColumn;
    private final String answerTextColumn;
    private final int qaSourceLimit;
    private final String vectorField;
    private final String questionTextField;
    private final String answerTextField;
    private final String companyIdField;
    private final int questionTextMaxLength;
    private final int answerTextMaxLength;
    private final int companyIdMaxLength;
    private final int milvusQueryLimit;
    private final int batchSize;
    private final HttpClient httpClient = HttpClient.newHttpClient();
    private final ObjectMapper objectMapper = new ObjectMapper();

    public QaController(
            JdbcTemplate jdbcTemplate,
            @Value("${milvus.host}") String milvusHost,
            @Value("${milvus.port}") Integer milvusPort,
            @Value("${milvus.collection-name}") String collectionName,
            @Value("${milvus.connect-timeout-seconds}") Integer milvusConnectTimeoutSeconds,
            @Value("${milvus.rpc-deadline-seconds}") Integer milvusRpcDeadlineSeconds,
            @Value("${embedding.endpoint}") String embeddingEndpoint,
            @Value("${embedding.api-key}") String embeddingApiKey,
            @Value("${qa-source.table-name}") String qaTableName,
            @Value("${qa-source.columns.qa-id}") String qaIdColumn,
            @Value("${qa-source.columns.company-id}") String companyIdColumn,
            @Value("${qa-source.columns.question}") String questionColumn,
            @Value("${qa-source.columns.answer-text}") String answerTextColumn,
            @Value("${qa-source.limit}") Integer qaSourceLimit,
            @Value("${milvus.fields.vector}") String vectorField,
            @Value("${milvus.fields.question-text}") String questionTextField,
            @Value("${milvus.fields.answer-text}") String answerTextField,
            @Value("${milvus.fields.company-id}") String companyIdField,
            @Value("${milvus.max-length.question-text}") Integer questionTextMaxLength,
            @Value("${milvus.max-length.answer-text}") Integer answerTextMaxLength,
            @Value("${milvus.max-length.company-id}") Integer companyIdMaxLength,
            @Value("${milvus.query-limit}") Integer milvusQueryLimit,
            @Value("${ingest.batch-size}") Integer batchSize
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.collectionName = requireText(collectionName, "milvus.collection-name");
        this.embeddingEndpoint = embeddingEndpoint == null ? "" : embeddingEndpoint;
        this.embeddingApiKey = embeddingApiKey == null ? "" : embeddingApiKey;
        this.qaTableName = requireText(qaTableName, "qa-source.table-name");
        this.qaIdColumn = requireText(qaIdColumn, "qa-source.columns.qa-id");
        this.companyIdColumn = requireText(companyIdColumn, "qa-source.columns.company-id");
        this.questionColumn = requireText(questionColumn, "qa-source.columns.question");
        this.answerTextColumn = requireText(answerTextColumn, "qa-source.columns.answer-text");
        this.qaSourceLimit = requirePositive(qaSourceLimit, "qa-source.limit");
        this.vectorField = requireText(vectorField, "milvus.fields.vector");
        this.questionTextField = requireText(questionTextField, "milvus.fields.question-text");
        this.answerTextField = requireText(answerTextField, "milvus.fields.answer-text");
        this.companyIdField = requireText(companyIdField, "milvus.fields.company-id");
        this.questionTextMaxLength = requirePositive(questionTextMaxLength, "milvus.max-length.question-text");
        this.answerTextMaxLength = requirePositive(answerTextMaxLength, "milvus.max-length.answer-text");
        this.companyIdMaxLength = requirePositive(companyIdMaxLength, "milvus.max-length.company-id");
        this.milvusQueryLimit = requirePositive(milvusQueryLimit, "milvus.query-limit");
        this.batchSize = requirePositive(batchSize, "ingest.batch-size");
        this.milvusClient = new MilvusServiceClient(
                ConnectParam.newBuilder()
                        .withHost(milvusHost)
                        .withPort(milvusPort)
                        .withConnectTimeout(
                                requirePositive(milvusConnectTimeoutSeconds, "milvus.connect-timeout-seconds"),
                                TimeUnit.SECONDS
                        )
                        .withRpcDeadline(
                                requirePositive(milvusRpcDeadlineSeconds, "milvus.rpc-deadline-seconds"),
                                TimeUnit.SECONDS
                        )
                        .build()
        );
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiError> handleException(Exception e) {
        String message = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
        return ResponseEntity.internalServerError().body(new ApiError(message));
    }

    @GetMapping("/qa")
    public List<QaData> getQaData() {
        return findQaData();
    }

    @GetMapping("/milvus-data")
    public List<Map<String, Object>> getMilvusData() {
        ensureCollectionExists();
        loadCollection();

        R<QueryResults> response = milvusClient.query(
                QueryParam.newBuilder()
                        .withCollectionName(collectionName)
                        .withExpr(buildAllRowsExpression())
                        .withOutFields(List.of(companyIdField, questionTextField, answerTextField))
                        .withLimit((long) milvusQueryLimit)
                        .build()
        );

        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Milvus data query failed: " + response.getMessage());
        }

        QueryResultsWrapper wrapper = new QueryResultsWrapper(response.getData());

        return wrapper.getRowRecords().stream()
                .map(row -> {
                    Map<String, Object> values = new LinkedHashMap<>(row.getFieldValues());
                    values.put("collection", collectionName);
                    return values;
                })
                .toList();
    }

    @GetMapping("/ingest")
    public String ingest() {
        List<QaData> qaList = findQaData();

        if (qaList.isEmpty()) {
            return "No QA data found in MySQL.";
        }

        ensureCollectionExists();
        clearCollection();

        int count = 0;

        for (int start = 0; start < qaList.size(); start += batchSize) {
            int end = Math.min(start + batchSize, qaList.size());
            List<QaData> batch = qaList.subList(start, end);

            List<String> texts = batch.stream()
                    .map(qa -> "Question: " + qa.question() + "\nAnswer: " + qa.answerText())
                    .toList();

            insertBatch(embedBatch(texts), batch);
            count += batch.size();
        }

        flushCollection("Milvus ingest flush failed");

        return "Milvus re-ingest completed: " + count + " rows";
    }

    private List<QaData> findQaData() {
        String sql = """
            SELECT
                %s AS qa_id,
                %s AS company_id,
                %s AS question,
                %s AS answer_text
            FROM %s
            LIMIT ?
        """.formatted(
                sqlIdentifier(qaIdColumn),
                sqlIdentifier(companyIdColumn),
                sqlIdentifier(questionColumn),
                sqlIdentifier(answerTextColumn),
                sqlIdentifier(qaTableName)
        );

        return jdbcTemplate.query(sql, ps -> ps.setInt(1, qaSourceLimit), (rs, rowNum) -> new QaData(
                rs.getLong("qa_id"),
                rs.getInt("company_id"),
                rs.getString("question"),
                rs.getString("answer_text")
        ));
    }

    private void ensureCollectionExists() {
        R<Boolean> response = milvusClient.hasCollection(
                HasCollectionParam.newBuilder()
                        .withCollectionName(collectionName)
                        .build()
        );

        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Milvus collection check failed: " + response.getMessage());
        }

        if (!Boolean.TRUE.equals(response.getData())) {
            throw new RuntimeException("Milvus collection does not exist: " + collectionName);
        }
    }

    private void loadCollection() {
        R<?> response = milvusClient.loadCollection(
                LoadCollectionParam.newBuilder()
                        .withCollectionName(collectionName)
                        .withSyncLoad(Boolean.TRUE)
                        .build()
        );

        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Milvus collection load failed: " + response.getMessage());
        }
    }

    private void clearCollection() {
        String deleteExpr = buildAllRowsExpression();

        R<?> response = milvusClient.delete(
                DeleteParam.newBuilder()
                        .withCollectionName(collectionName)
                        .withExpr(deleteExpr)
                        .build()
        );

        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Milvus collection clear failed: " + response.getMessage());
        }

        flushCollection("Milvus delete flush failed");
    }

    private String buildAllRowsExpression() {
        R<DescribeCollectionResponse> response = milvusClient.describeCollection(
                DescribeCollectionParam.newBuilder()
                        .withCollectionName(collectionName)
                        .build()
        );

        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Milvus collection describe failed: " + response.getMessage());
        }

        FieldSchema primaryKey = response.getData().getSchema().getFieldsList().stream()
                .filter(FieldSchema::getIsPrimaryKey)
                .findFirst()
                .orElseThrow(() -> new RuntimeException("Milvus primary key field not found: " + collectionName));

        return switch (primaryKey.getDataType()) {
            case Int8, Int16, Int32, Int64 -> primaryKey.getName() + " >= 0";
            case String, VarChar -> primaryKey.getName() + " != \"\"";
            default -> throw new RuntimeException(
                    "Unsupported Milvus primary key type for delete: " + primaryKey.getDataType()
            );
        };
    }

    private void flushCollection(String failureMessage) {
        R<?> response = milvusClient.flush(
                FlushParam.newBuilder()
                        .addCollectionName(collectionName)
                        .withSyncFlush(Boolean.TRUE)
                        .build()
        );

        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException(failureMessage + ": " + response.getMessage());
        }
    }

    private void insertBatch(List<List<Float>> vectors, List<QaData> qaList) {
        if (vectors.size() != qaList.size()) {
            throw new RuntimeException("Vector count does not match QA count.");
        }

        List<String> questionTexts = new ArrayList<>();
        List<String> answerTexts = new ArrayList<>();
        List<String> companyIds = new ArrayList<>();

        for (QaData qa : qaList) {
            questionTexts.add(cut(qa.question(), questionTextMaxLength));
            answerTexts.add(cut(qa.answerText(), answerTextMaxLength));
            companyIds.add(cut(String.valueOf(qa.companyId()), companyIdMaxLength));
        }

        List<InsertParam.Field> fields = List.of(
                new InsertParam.Field(vectorField, vectors),
                new InsertParam.Field(questionTextField, questionTexts),
                new InsertParam.Field(answerTextField, answerTexts),
                new InsertParam.Field(companyIdField, companyIds)
        );

        R<?> response = milvusClient.insert(
                InsertParam.newBuilder()
                        .withCollectionName(collectionName)
                        .withFields(fields)
                        .build()
        );

        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Milvus batch insert failed: " + response.getMessage());
        }
    }

    private List<List<Float>> embedBatch(List<String> texts) {
        try {
            String endpoint = requireText(embeddingEndpoint, "embedding.endpoint");
            String apiKey = requireText(embeddingApiKey, "embedding.api-key");
            String requestBody = objectMapper.writeValueAsString(new EmbeddingBatchRequest(texts));

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(endpoint))
                    .header("Content-Type", "application/json")
                    .header("X-Api-Key", apiKey)
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString()
            );

            if (response.statusCode() != 200) {
                throw new RuntimeException(
                        "Embedding API batch request failed: "
                                + response.statusCode()
                                + " / "
                                + response.body()
                );
            }

            JsonNode root = objectMapper.readTree(response.body());

            if (!root.has("data")) {
                throw new RuntimeException("Unexpected embedding response: " + response.body());
            }

            List<List<Float>> result = new ArrayList<>();

            for (JsonNode item : root.get("data")) {
                JsonNode embeddingNode = item.get("embedding");
                List<Float> vector = new ArrayList<>();

                for (JsonNode value : embeddingNode) {
                    vector.add((float) value.asDouble());
                }

                result.add(vector);
            }

            return result;
        } catch (Exception e) {
            throw new RuntimeException("Embedding batch creation failed.", e);
        }
    }

    private String cut(String value, int maxLength) {
        if (value == null) {
            return "";
        }

        return value.length() > maxLength
                ? value.substring(0, maxLength)
                : value;
    }

    private String sqlIdentifier(String value) {
        String text = requireText(value, "SQL identifier");
        String[] parts = text.split("\\.");
        List<String> quotedParts = new ArrayList<>();

        for (String part : parts) {
            if (!part.matches("[A-Za-z0-9_]+")) {
                throw new IllegalArgumentException("Invalid SQL identifier: " + value);
            }

            quotedParts.add("`" + part + "`");
        }

        return String.join(".", quotedParts);
    }

    private String requireText(String value, String propertyName) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(propertyName + " must not be blank.");
        }

        return value;
    }

    private int requirePositive(Integer value, String propertyName) {
        if (value == null || value <= 0) {
            throw new IllegalArgumentException(propertyName + " must be greater than 0.");
        }

        return value;
    }
}

record QaData(Long qaId, Integer companyId, String question, String answerText) {
}

record EmbeddingBatchRequest(List<String> input) {
}

record ApiError(String message) {
}
