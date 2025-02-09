import os
import boto3
from botocore.config import Config
from typing import Dict, List, Optional
import datetime as dt
from pathlib import Path
import gzip
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from connectors.supabase_connector import SupabaseConnector
from utils.logging_config import logger


class PolygonDataDownloader:
    def __init__(
        self, access_key: str, secret_key: str, supabase_connector, max_workers: int = 4
    ):
        """
        Initialize Polygon.io data downloader

        Args:
            access_key: Polygon.io access key
            secret_key: Polygon.io secret key
            supabase_connector: Instance of SupabaseConnector
            max_workers: Maximum number of concurrent downloads
        """
        self.max_workers = max_workers
        self.supabase = supabase_connector

        # Initialize Polygon.io S3 client
        self.s3 = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ).client(
            "s3",
            endpoint_url="https://files.polygon.io",
            config=Config(signature_version="s3v4"),
        )

        # Create temp directory for downloads
        self.temp_dir = Path("./temp_downloads")
        self.temp_dir.mkdir(exist_ok=True)

    def _get_date_range(
        self, years: Optional[int] = None, days: Optional[int] = None
    ) -> tuple[dt.datetime, dt.datetime]:
        """
        Get date range for historical data

        Args:
            years: Number of years to look back (default: None)
            days: Number of days to look back (default: None)

        Returns:
            Tuple of (start_date, end_date)

        Raises:
            ValueError: If neither years nor days is provided, or if both are provided
        """
        if years is None and days is None:
            raise ValueError("Must provide either years or days")
        if years is not None and days is not None:
            raise ValueError("Cannot provide both years and days, choose one")

        end_date = dt.datetime.now()

        if years is not None:
            start_date = end_date - dt.timedelta(days=years * 365)
        else:
            start_date = end_date - dt.timedelta(days=days)

        return start_date, end_date

    def _get_available_files(
        self, start_date: dt.datetime, end_date: dt.datetime
    ) -> List[str]:
        """Get list of available files within date range"""
        paginator = self.s3.get_paginator("list_objects_v2")
        files = []
        print("Fetching files")
        for page in paginator.paginate(
            Bucket="flatfiles", Prefix="us_stocks_sip/trades_v1/"
        ):  
            print("Fetching files 1")
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                print("Fetching files 2")
                key = obj["Key"]
                try:
                    # Extract date from file path
                    date_str = key.split("/")[-1].split(".")[0]
                    file_date = dt.datetime.strptime(date_str, "%Y-%m-%d")

                    if start_date <= file_date <= end_date:
                        files.append(key)
                except (ValueError, IndexError):
                    continue
        print("Fetching files 3")
        return files

    def _process_file(self, file_key: str, bucket_name: str) -> Dict:
        """Process a single file"""
        try:
            # Download file
            local_path = self.temp_dir / file_key.split("/")[-1]
            self.s3.download_file("flatfiles", file_key, str(local_path))

            # Upload to Supabase storage
            upload_path = f"historical/{file_key.split('/')[-1]}"
            with open(local_path, "rb") as f:
                result = self.supabase.save_to_storage(
                    {
                        "content": f.read(),
                        "path": upload_path,
                        "content_type": "application/gzip",
                    },
                    bucket_name,
                )

            # Clean up
            local_path.unlink()

            return {
                "success": True,
                "file": file_key,
                "upload_path": upload_path,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Error processing file {file_key}: {str(e)}")
            return {"success": False, "file": file_key, "error": str(e)}

    def download_historical_data(
        self,
        bucket_name: str = "historical-data",
        years: Optional[int] = None,
        days: Optional[int] = None,
    ) -> Dict:
        """
        Download historical stock data from Polygon.io and upload to Supabase

        Args:
            bucket_name: Name of the Supabase storage bucket
            years: Number of years of historical data to fetch
            days: Number of days of historical data to fetch

        Returns:
            Dict with operation results
        """
        try:
            # Get date range
            start_date, end_date = self._get_date_range(years=years, days=days)
            logger.info(f"Fetching data from {start_date} to {end_date}")

            # Get available files
            files = self._get_available_files(start_date, end_date)
            print(files)
            logger.info(f"Found {len(files)} files to process")

            if not files:
                return {
                    "success": False,
                    "error": "No files found for the specified date range",
                }

            # Process files using ThreadPoolExecutor
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._process_file, file_key, bucket_name)
                    for file_key in files
                ]

                for future in futures:
                    result = future.result()
                    results.append(result)

            # Compile statistics
            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]

            return {
                "success": True,
                "total_files": len(files),
                "successful_uploads": len(successful),
                "failed_uploads": len(failed),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "results": results,
            }

        except Exception as e:
            logger.error(f"Error in download pipeline: {str(e)}")
            return {"success": False, "error": str(e)}


supabase = SupabaseConnector(
    supabase_url=os.getenv("SUPABASE_URL", ""),
    supabase_key=os.getenv("SUPABASE_KEY", ""),
)

downloader = PolygonDataDownloader(
    access_key=os.getenv("POLYGON_S3_KEY_ID", ""),
    secret_key=os.getenv("POLYGON_S3_ACCESS_KEY", ""),
    supabase_connector=supabase,
    max_workers=4,
)

# Download last 30 days of data
result = downloader.download_historical_data(bucket_name="historical-data", days=5)

# Or download 2 years of data
# result = downloader.download_historical_data(bucket_name="historical-data", years=2)

# Check results
if result["success"]:
    print(f"Successfully processed {result['successful_uploads']} files")
    print(f"Failed to process {result['failed_uploads']} files")
else:
    print(f"Download failed: {result['error']}")
