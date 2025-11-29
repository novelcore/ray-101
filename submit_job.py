"""
Submit Ray Job to cluster via Jobs API
This works around version mismatch issues by using the HTTP API
"""
import requests
import json
import time
import sys

def submit_ray_job(job_name, entrypoint, runtime_env=None):
    """Submit a job to Ray cluster via Jobs API"""
    jobs_url = "http://localhost:8266/api/jobs"
    
    payload = {
        "entrypoint": entrypoint,
        "job_id": job_name,
        "runtime_env": runtime_env or {}
    }
    
    print(f"Submitting job '{job_name}' to Ray cluster...")
    response = requests.post(jobs_url, json=payload)
    
    if response.status_code == 200:
        print(f"✅ Job submitted successfully!")
        return response.json()
    else:
        print(f"❌ Failed to submit job: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def get_job_status(job_id):
    """Get status of a Ray job"""
    jobs_url = f"http://localhost:8266/api/jobs/{job_id}"
    response = requests.get(jobs_url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

if __name__ == "__main__":
    # For now, this is a template
    # The actual job submission would need the pipeline code to be accessible
    print("Ray Jobs API submission script")
    print("Note: This requires the pipeline to be accessible from the cluster")
    print("Dashboard: http://localhost:8266")

