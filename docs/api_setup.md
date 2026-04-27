# API Setup

The pipeline uses five external APIs. All are free to register for
academic non-commercial use; full-text retrieval typically requires your
institution to hold the relevant subscription or to have signed the
publisher's text-and-data-mining (TDM) agreement.

> **Compliance.** This pipeline downloads full-text content only through
> publisher-provided APIs (Wiley TDM, Elsevier ScienceDirect, Springer
> Nature, plus Unpaywall for openly-licensed copies). 
> You are responsible for verifying that your usage complies with:
> (1) each publisher's API terms of service, (2) your institution's TDM
> agreement, and (3) any applicable copyright law in your jurisdiction.

Copy `.env.example` to `.env` and paste each key into the matching variable.

## Wiley TDM

**Registration:** <https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining>

You need:

* An ORCID iD (<https://orcid.org>)
* To accept Wiley's click-through TDM license
* An institutional IP for full-text retrieval

Wiley returns full PDFs reliably when your institution has a Wiley
subscription. The token is tied to your ORCID — do not share it.

**.env variable:** `WILEY_TDM_TOKEN`

## Elsevier (ScienceDirect)

**Registration:** <https://dev.elsevier.com/apikey/manage>

You need:

* An Elsevier developer account
* Your request to be made from an institutional IP range that has a
  ScienceDirect subscription

Without a subscription, the API returns abstract-only content. With one, it
returns full-text PDFs.

**.env variable:** `ELSEVIER_API_KEY`

## Springer Nature (two APIs)

**Registration:** <https://dev.springernature.com/>

Two separate keys:

* **Open Access API** — returns full text for Springer OA content
* **Meta API** — metadata + PDF URLs for the wider Springer catalog

Free tier is sufficient for most academic use cases (approx. 5,000
requests/day).

**.env variables:** `SPRINGER_OA_API_KEY`, `SPRINGER_META_API_KEY`

## OpenAI (for knowledge graph extraction)

**Registration:** <https://platform.openai.com/api-keys>

Required only for script 05 (knowledge graph extraction). The default model
is `gpt-4o-mini`. You can change this to use other models.

**.env variable:** `OPENAI_API_KEY`

## Unpaywall (no key needed)

Unpaywall is used as an automatic fallback for DOIs where the publisher APIs
don't return a full PDF. It only needs an email address for its rate-limit
policy — set this in your `.env`:

**.env variable:** `CONTACT_EMAIL`

## ACS (American Chemical Society)

The pipeline can fetch ACS PDFs through their DOI-based PDF endpoint when
the request originates from an IP authenticated under an institutional
ACS subscription (no API key required). Verify that your institution's
ACS subscription explicitly permits programmatic / text-and-data-mining
use before running this downloader at scale; some institutions limit
automated access or require a separate TDM addendum.

## Other publishers

For publishers not listed above (IOP, Taylor & Francis, SAGE, Emerald,
RSC, etc.), the pipeline falls back to Unpaywall and logs everything else
as a failure for manual follow-up (interlibrary loan, institutional
proxy in a browser, contacting the author for a preprint, etc.).
