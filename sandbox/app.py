import json
from math import ceil
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sandbox.src.state import get_experiment_data

app = FastAPI()

app.mount("/static", StaticFiles(directory="sandbox/static"), name="static")
templates = Jinja2Templates(directory="sandbox/templates")

PRODUCTS_PER_PAGE = 30
DATASET_DIR = '../datasets'

def load_products_for_query(query):
    """Load products from in-memory experiment data or fallback to JSON file"""
    # First try to get experiment data from memory
    experiment_data = get_experiment_data()
    if experiment_data:
        return experiment_data
    
    # Fallback to loading from JSON file
    filename = query.lower().replace(' ', '+') + '.json'
    json_file = os.path.join(DATASET_DIR, filename)
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
            print(f"Successfully loaded {len(products)} products from file")
            return products
    except FileNotFoundError:
        print(f"No product file found for: {filename}")
        print(f"Make sure the file exists at: {json_file}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {filename}: {e}")
        return []

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = "", sort: str = "similarity", page: int = 1):
    query = q
    sort_by = sort
    
    if query:
        # Load products from corresponding JSON file
        all_results = load_products_for_query(query)

        # Sort products if needed
        if sort_by == 'price_asc':
            all_results.sort(key=lambda x: float(x.get('price', '0').replace('$', '').replace(',', '')))
        elif sort_by == 'price_desc':
            all_results.sort(key=lambda x: float(x.get('price', '0').replace('$', '').replace(',', '')), reverse=True)
        
        # Calculate pagination info
        total_products = len(all_results)
        total_pages = ceil(total_products / PRODUCTS_PER_PAGE)
        
        # Get products for current page
        start_idx = (page - 1) * PRODUCTS_PER_PAGE
        end_idx = start_idx + PRODUCTS_PER_PAGE
        current_products = all_results[start_idx:end_idx]
    else:
        current_products = []
        total_pages = 0
        page = 1
    
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "query": query,
            "products": current_products,
            "current_sort": sort_by,
            "current_page": page,
            "total_pages": total_pages
        }
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000, reload=True)