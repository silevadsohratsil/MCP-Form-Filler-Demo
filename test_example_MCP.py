import asyncio
from fastmcp import Client

async def test_login():
    # Create a client that connects to your MCP server
    client = Client("mcp_form_filler.py")
    
    async with client:
        result = await client.call_tool(
            "fill_form_and_check",
            {
                "url": "http://localhost:3000/login",
                "fields": {
                    "Email": "test@example.com",
                    "Password": "secret123"
                },
                "submit_text": "Sign in",
                "must_contain_text": "Welcome back",
                "headless": True
            }
        )
        
        print(f"Test Result: {result['result']}")
        print(f"Details: {result['final']}")
        
        # Check if test passed
        if result['result'] == 'PASS':
            print("OK! -  Login test passed!")
        else:
            print("FAIL! - Login test failed!")
            print("Steps taken:")
            for step in result['steps']:
                print(f"  - {step}")

# Run the test
asyncio.run(test_login())
