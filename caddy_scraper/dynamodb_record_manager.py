import boto3
from botocore.exceptions import ClientError, NoRegionError, NoCredentialsError
from typing import List, Optional, Sequence, Dict
from decimal import Decimal
import time
import asyncio
from langchain_core.indexing import RecordManager


class DynamoDBRecordManager(RecordManager):
    def __init__(
        self,
        namespace: str,
        table_name: str,
        region_name: Optional[str] = None,
    ) -> None:
        super().__init__(namespace=namespace)
        self.table_name = table_name
        self.audit_table_name = f"{table_name}-audit"

        try:
            session = boto3.Session(region_name=region_name)

            if session.region_name is None:
                raise NoRegionError

            session.get_credentials().get_frozen_credentials()

            self.dynamodb = session.resource("dynamodb")
            self.table = self.dynamodb.Table(table_name)
            self.audit_table = self.dynamodb.Table(self.audit_table_name)

            # Check if tables exist, create if they don't
            self._ensure_table_exists(self.table, self.create_schema)
            self._ensure_table_exists(
                self.audit_table, self.create_audit_schema)

        except NoRegionError:
            raise ValueError(
                "No AWS region specified. Please configure your AWS region."
            )
        except NoCredentialsError:
            raise ValueError(
                "No valid AWS credentials found. Please configure your AWS credentials."
            )

    def _ensure_table_exists(self, table, create_method):
        try:
            table.table_status
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                create_method()
            else:
                raise

    def create_schema(self) -> None:
        try:
            table = self.dynamodb.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {"AttributeName": "namespace", "KeyType": "HASH"},
                    {"AttributeName": "key", "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "namespace", "AttributeType": "S"},
                    {"AttributeName": "key", "AttributeType": "S"},
                    {"AttributeName": "group_id", "AttributeType": "S"},
                    {"AttributeName": "updated_at", "AttributeType": "N"},
                ],
                GlobalSecondaryIndexes=[
                    {
                        "IndexName": "GroupIdIndex",
                        "KeySchema": [
                            {"AttributeName": "group_id", "KeyType": "HASH"},
                            {"AttributeName": "updated_at", "KeyType": "RANGE"},
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                    }
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            table.wait_until_exists()
            self.table = table
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceInUseException":
                raise

    def create_audit_schema(self) -> None:
        try:
            table = self.dynamodb.create_table(
                TableName=self.audit_table_name,
                KeySchema=[
                    {"AttributeName": "namespace", "KeyType": "HASH"},
                    {"AttributeName": "timestamp", "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "namespace", "AttributeType": "S"},
                    {"AttributeName": "timestamp", "AttributeType": "N"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            table.wait_until_exists()
            self.audit_table = table
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceInUseException":
                raise

    def log_audit(self, completion: Dict[str, int], success: bool) -> None:
        timestamp = int(time.time() * 1000)
        item = {
            "namespace": self.namespace,
            "timestamp": timestamp,
            "success": success,
            "num_added": completion.get("num_added", 0),
            "num_updated": completion.get("num_updated", 0),
            "num_skipped": completion.get("num_skipped", 0),
            "num_deleted": completion.get("num_deleted", 0),
        }
        self.audit_table.put_item(Item=item)

    async def acreate_schema(self) -> None:
        await asyncio.to_thread(self.create_schema)

    def get_time(self) -> float:
        return time.time()

    async def aget_time(self) -> float:
        return await asyncio.to_thread(self.get_time)

    def update(
        self,
        keys: Sequence[str],
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        update_time = self.get_time()
        if time_at_least and update_time < time_at_least:
            raise AssertionError(f"Time sync issue: {
                                 update_time} < {time_at_least}")

        if group_ids is None:
            group_ids = [None] * len(keys)

        with self.table.batch_writer() as batch:
            for key, group_id in zip(keys, group_ids):
                item = {
                    "namespace": self.namespace,
                    "key": key,
                    "updated_at": Decimal(str(update_time)),
                }
                if group_id:
                    item["group_id"] = group_id
                batch.put_item(Item=item)

    async def aupdate(
        self,
        keys: Sequence[str],
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        await asyncio.to_thread(self.update, keys, group_ids, time_at_least)

    def exists(self, keys: Sequence[str]) -> List[bool]:
        results = []
        for key in keys:
            try:
                response = self.table.get_item(
                    Key={"namespace": self.namespace, "key": key}
                )
                results.append("Item" in response)
            except ClientError:
                results.append(False)
        return results

    async def aexists(self, keys: Sequence[str]) -> List[bool]:
        return await asyncio.to_thread(self.exists, keys)

    def list_keys(
        self,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        if group_ids:
            response = self.table.query(
                IndexName="GroupIdIndex",
                KeyConditionExpression="group_id = :gid",
                ExpressionAttributeValues={":gid": group_ids[0]},
                ScanIndexForward=False,
            )
        else:
            response = self.table.query(
                KeyConditionExpression="namespace = :ns",
                ExpressionAttributeValues={":ns": self.namespace},
                ScanIndexForward=False,
            )

        items = response["Items"]
        keys = [item["key"] for item in items]

        if after:
            keys = [
                k for k, item in zip(keys, items) if float(item["updated_at"]) > after
            ]
        if before:
            keys = [
                k for k, item in zip(keys, items) if float(item["updated_at"]) < before
            ]
        if limit:
            keys = keys[:limit]

        return keys

    async def alist_keys(
        self,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        return await asyncio.to_thread(self.list_keys, before, after, group_ids, limit)

    def delete_keys(self, keys: Sequence[str]) -> None:
        with self.table.batch_writer() as batch:
            for key in keys:
                batch.delete_item(
                    Key={"namespace": self.namespace, "key": key})

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        await asyncio.to_thread(self.delete_keys, keys)
