# Issue tracker

**Tracker:** GitHub Issues
**Repo:** `oslumen/stratix`
**CLI:** `gh issue`

## Operations

- **Create:** `gh issue create --title "..." --body "..." --label "needs-triage" --repo oslumen/stratix`
- **List:** `gh issue list --repo oslumen/stratix --label "ready-for-agent" --limit 50 --json number,title,labels`
- **View:** `gh issue view <number> --repo oslumen/stratix --json number,title,body,labels,comments`
- **Update labels:** `gh issue edit <number> --add-label "..." --remove-label "..." --repo oslumen/stratix`
- **Close:** `gh issue close <number> --repo oslumen/stratix`

## Automation

- GitHub Actions workflow `.github/workflows/test.yml` runs on push and PR.
- Code review happens via PR review, not issue comments.
