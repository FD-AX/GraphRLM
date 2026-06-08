from neo4j import GraphDatabase


class Neo4jClient:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        connection_timeout: float = 3.0,
        max_transaction_retry_time: float = 3.0,
    ):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            connection_timeout=connection_timeout,
            max_transaction_retry_time=max_transaction_retry_time,
        )

    def close(self) -> None:
        self.driver.close()

    def execute_write(self, query: str, parameters: dict | None = None) -> None:
        parameters = parameters or {}

        with self.driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(query, **parameters).consume()
            )

    def execute_read(self, query: str, parameters: dict | None = None) -> list[dict]:
        parameters = parameters or {}

        with self.driver.session() as session:
            result = session.execute_read(
                lambda tx: list(tx.run(query, **parameters))
            )

        return [record.data() for record in result]
