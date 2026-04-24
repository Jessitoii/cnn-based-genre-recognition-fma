# Web Application Frontend

This directory contains the user interface module that communicates contextually with the generic categorization inference API securely engineered via Next.js.

## Frontend Architecture
The frontend leverages the robust app-router orientation of Next.js 14+ mapped alongside responsive, dynamic UI utility variables exposed through Tailwind CSS framing native desktop/mobile viewport scales gracefully. 

As a predominantly componentized state container, it processes complex browser file buffer selections orchestrating background HTTP operations towards `localhost:8000/predict` and asynchronously visualizes incoming probabilistic json payload vectors onto animated UI grids. 

## Components Directory
* `src/components`: Encompasses modular, reusable React segments.
* `src/app`: Primary Next.js root layout structures and routing handlers governing initial client rendering phases logically across separate UI contexts.

## How to Run

1. Make sure you have Node > 18.x standard configured.
2. Ensure the adjacent Backend Service `localhost:8000` is persistently executing via another terminal concurrently.
3. Use your CLI of choice mapping to package deployments:

```bash
npm install
# Setup dependencies 

npm run dev
# Init development node
```
4. Access `http://localhost:3000` iteratively checking inference states via direct `.mp3` drop interfaces locally inside the browser.
