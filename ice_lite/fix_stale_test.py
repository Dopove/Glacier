import os
import asyncio
from pathlib import Path
from ice_lite.sdk import init as init_ice_lite
import uuid

async def fix_test_stale_knowledge():
    ice = await init_ice_lite()
    session_id = f"stale-test-{uuid.uuid4()}"
    
    # Create temporary files for old and new policies
    temp_dir = Path("./temp_policies")
    temp_dir.mkdir(exist_ok=True)

    old_policy_path = temp_dir / "old_policy.txt"
    new_policy_path = temp_dir / "new_policy.txt"

    # Write content to temporary files
    old_policy_content = "Old Policy: Remote work is forbidden."
    new_policy_content = "New Policy: Remote work is encouraged."
    
    old_policy_path.write_text(old_policy_content)
    new_policy_path.write_text(new_policy_content)

    # Ingest old (stale) doc
    old_time = datetime.now() - timedelta(days=365)
    await ice.ingest(str(old_policy_path), x_session_id=session_id, 
                    metadata={"created_at": old_time.isoformat(), "valid_until": (old_time + timedelta(days=30)).isoformat()})
    
    # Ingest new (valid) doc
    await ice.ingest(str(new_policy_path), x_session_id=session_id)

    # Clean up temporary files and directory
    old_policy_path.unlink()
    new_policy_path.unlink()
    temp_dir.rmdir()

    print("Temporary files created, ingested, and cleaned up.")

if __name__ == "__main__":
    from datetime import datetime, timedelta # Import here for the temporary script
    asyncio.run(fix_test_stale_knowledge())
