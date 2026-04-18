# Contributing to KG-MAG

Thanks for your interest in contributing.

## Prerequisites

- Git
- Docker and Docker Compose
- Python 3.11+
- Node.js 20+

## Local Setup

1. Fork the repository and clone your fork.
2. Copy environment variables:
   - `cp .env.example .env`
3. Fill required API keys in `.env`.

Run full stack with Docker:

```bash
docker compose up --build
```

App URLs:

- Frontend: http://localhost:3000
- Backend: http://localhost:8080
- Health check: http://localhost:8080/health

## Branch Naming

Use one of these patterns:

- `feature/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`
- `refactor/<short-description>`

Example:

```bash
git checkout -b feature/improve-retrieval-ranking
```

## Development Workflow

1. Create a branch from `main`.
2. Make focused changes.
3. Add or update tests for behavior changes.
4. Run checks locally.
5. Push your branch and open a pull request.

## Testing Expectations

Run backend checks before opening a PR:

```bash
pytest tests/ -v
pytest tests/ --cov=backend --cov=ingestion --cov-report=term-missing
```

Run frontend checks:

```bash
cd frontend
npm ci
npm run lint
npm run build
```

## Pull Request Checklist

- [ ] Linked related issue (or explained why not needed)
- [ ] Added tests or justified why not needed
- [ ] Updated docs for behavior/config changes
- [ ] Verified Docker workflow still works
- [ ] Kept changes scoped to one concern

## Reporting Bugs

Please use the Bug Report template and include:

- Clear reproduction steps
- Expected vs actual behavior
- Logs and stack traces
- Environment details

## Need Help?

- Open a GitHub Discussion for design or usage questions.
- Open a GitHub Issue for confirmed bugs or actionable enhancements.
