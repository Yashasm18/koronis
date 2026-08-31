# Security Policy

## Scope

Koronis is a **defence-only research prototype**. It is not a production system and
handles no sensitive data:

- No real cardholder data, no real BIN ranges, no live payment integration.
- No network capability anywhere in the package — it operates only on in-memory
  dataframes. This is checked by a test and reproducible with one command:

  ```bash
  grep -rnE "^(import|from) (requests|urllib|socket|http|aiohttp|subprocess)" koronis/
  ```

- The campaign generator exists solely to produce labelled test data. It reproduces only
  attack characteristics already documented publicly in Visa's anti-enumeration guidance
  and cannot be pointed at any external system.

## Supported versions

Only the `main` branch is maintained. There are no tagged releases.

## Reporting a vulnerability

If you find a security issue in the code or its dependencies:

1. **Preferred:** open a private report via GitHub's
   [Security Advisories](https://github.com/Yashasm18/koronis/security/advisories/new)
   ("Report a vulnerability").
2. **Alternative:** email `sssyashu850@gmail.com` with the details and, if possible, a
   minimal reproduction.

Please do not open a public issue for anything that could be exploited.

This is a student project maintained by one person, so responses are best-effort. Expect
an acknowledgement within about a week.

## Out of scope

- The synthetic campaign generator producing attack-shaped data is by design and clearly
  labelled; it is not a vulnerability.
- Model evasion (an attacker spreading a campaign across enough fresh infrastructure to
  leave no graph signal) is a documented limitation and an open research question, not a
  defect. See the "Limitations" section of the README.
