import io
import pandas as pd
import boto3


class s3_operations:
    def __init__(self, bucket_name: str, access_key: str, secret_key: str, region_name: str = 'eu-north-1'):
        self.bucket_name = bucket_name
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name,
        )

    def fetch_file_from_s3(self, key: str) -> pd.DataFrame:
        obj = self.s3.get_object(Bucket=self.bucket_name, Key=key)
        body = obj['Body'].read()
        return pd.read_csv(io.BytesIO(body))
