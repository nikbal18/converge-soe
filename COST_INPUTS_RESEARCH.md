# Cost inputs: what is sourced, what is not

Research pass 2026-09-17, against the six items in §12 of
`Cost_Translation_Section.docx`. Every figure below is public and cited. Items
marked NOT SOURCED are still open and say exactly where to look.

---

## 1. Wholesale price — SOURCED, but the shape is wrong

**NSW, FY2023-24 average: $114/MWh.** AER *State of the energy market 2024*,
Chapter 2, Figure 2.4 (p.19), a 27% decline on the prior year. This is the year
your simulation windows sit in (Dec 2023 summer week, Jun-Jul 2023 winter week),
so it is the right vintage to use.

For context on where the market has gone since: NSW averaged **$75/MWh in
Q2 2026**, down $86/MWh (-53%) year on year, against a NEM average of $74/MWh
(AEMO *Quarterly Energy Dynamics Q2 2026*, §2.2).

**Your placeholder of $0.10/kWh is $100/MWh, which is within 15% of the FY2023-24
NSW average. So the level is defensible. The problem is the shape.**

The export the DTR recovers is midday PV export. Midday is precisely when NSW
wholesale prices are lowest and increasingly negative: negative price intervals
in NSW rose **61%** in 2023-24 (same report, Figure 2.6, p.22), and by Q2 2026
negative prices still occurred in 3.3% of NSW intervals. Valuing recovered
midday export at a flat annual average therefore **overstates its wholesale
value**, possibly by a lot.

This is not a nuisance, it is a result. It means:

- to the **customer**, recovered export is worth their feed-in tariff
- to the **market**, the same energy may be worth close to zero or negative
- those are different numbers for the same kWh, which is exactly the stakeholder
  split §9.3 already argues

**Recommendation:** keep the flat price as the headline for comparability, but
compute a second figure using the time-weighted NSW spot price over your actual
simulated intervals. You have the timestamps. AEMO publishes 30-minute NSW1
prices, and your intervals are already 30 minutes. That converts a caveat into
a quantified finding, and it is a genuinely novel one for a DOE paper.

---

## 2. Emissions intensity — PARTLY SOURCED, and the marginal version does not exist

**Average (location-based) scope 2 factor, NSW and ACT: 0.64 kg CO2-e/kWh**,
National Greenhouse Accounts Factors 2025. NSW and the ACT share a single
factor.

**Verify this yourself from the primary PDF.** `dcceew.gov.au` blocks automated
fetching, so the figure above came from a secondary compilation. Download
*Australian National Greenhouse Accounts Factors 2025* from DCCEEW and read the
scope 2 table directly. Get the scope 3 factor from the same table while you are
there; the only scope 3 figure I could confirm was a national residual-mix value
and it does not apply to NSW.

**The marginal emissions intensity is not published by anyone.** NGA publishes
average factors only, and the NSW government's own reporting guidelines use
average, location-based factors and do not discuss the marginal case at all. So
there is no citable Australian marginal factor to drop in.

You have three honest options:

1. Use the average factor and state plainly that it is an average, so the
   emissions figure is indicative and probably conservative.
2. Assume the marginal generator and justify it (NSW midday marginal plant is
   usually black coal or CCGT, and the emissions intensity of each is
   published). State the assumption in the open.
3. Drop the emissions channel from the headline and report it as a scoping
   note.

**The ACT complication you must address either way.** The ACT government
procures 100% renewable electricity through its reverse auctions. Attributing
avoided emissions to additional ACT rooftop export is therefore contestable, and
an examiner from the ACT will raise it. Decide your position, say it in one
sentence, and move on.

---

## 3. Carbon value — SOURCED

**ACCU generic spot: $36.28/tCO2-e. Safeguard Mechanism Credit spot:
$36.40/tCO2-e.** Both as at 31 March 2026, from the Clean Energy Regulator's
*Quarterly Carbon Market Report* for the December/March quarter.

Verify directly at the CER quarterly carbon market reports page before citing.
The CER site did not respond to an automated fetch, so the figures above came
via a market data aggregator quoting that report.

Note this is a **compliance market price**, not a social cost of carbon. If your
framing is "what would this abatement be worth to a liable entity", the ACCU
price is right. If your framing is "what is this worth to society", it is the
wrong instrument and you want a published social cost of carbon instead. §2.4 of
your draft is titled "Society", so decide which one §7 is actually pricing.

---

## 4. ACT feed-in tariff — SOURCED, but there is no single number

**There is no regulated ACT feed-in tariff.** The ACT Government states directly
that "these tariffs are voluntary, and the rates are not regulated in the ACT".
Retailer offers as at September 2026 span roughly **5c to 24.6c/kWh**, with the
high end attached to conditional or limited-volume offers rather than a standard
rate.

**The legacy ACT Premium Feed-in Tariff** (Electricity Feed-in (Renewable Energy
Premium) Act 2008) closed to new entrants on **13 July 2011**, pays for 20 years
from connection, and still covered about **9,895 generators totalling 34.8 MW**
in 2024-25. Those 20-year terms are now expiring, which matters: a customer
rolling off a premium tariff onto a 5-8c offer faces a step change in what
curtailment costs them.

**Recommendation:** do not pick one rate. Use a low/central/high band and run it
through the existing sensitivity machinery, and state that the ACT is unregulated
so the value of recovered export to a customer depends on their retailer. The
legacy scheme is worth a sentence in limitations because a minority of your
modelled customers may still be on it.

---

## 5. Unit replacement cost — NOT SOURCED for Evoenergy, and your placeholder looks high

**Your `unit_replacement_cost_flat_aud: 100000` is probably 2 to 5 times too
high.** Public comparators for *distribution* transformers:

| Source | Figure |
|---|---|
| Ergon Energy, *Distribution Transformer Replacements Business Case*, Jan 2024, §4.2.2 | **$13,200** (defective pole-mounted) to **$47,200** (failure replacement, large pole-mounted) |
| Powercor/CitiPower, *Asset class overview: distribution transformers* (PAL BUS 4.06), Jan 2025, Table 5 | approx **$19,500** per kiosk transformer, implied from volumes and expenditure |

Your fleet is ACT substations, which are kiosk and ground-mounted rather than
pole-mounted, and those cost more than pole-mounted units. Evoenergy's own
capital expenditure proposal says its "transformer unit costs are comparable to
the NEM median in most categories excluding pole and kiosk mounted transformers
where Evoenergy costs are higher". So you should land above the Powercor kiosk
figure, but $100,000 is a long way above anything in the public record.

This matters more than any other parameter here, because unit cost scales the
entire network-side result linearly.

### Where the real number is

The AER **Category Analysis RIN** annual responses publish per-DNSP unit costs by
asset category, including distribution transformers. That is the dataset behind
the "comparable to the NEM median" claim, and it is public, but it lives in
spreadsheets on the AER site rather than in a PDF you can cite a page of. Look
for Evoenergy's Category Analysis RIN response, or the AER's economic
benchmarking RIN data set.

### Two corrections to your `cost_params.yaml`

1. **The reference is wrong.** The file points at "Evoenergy Appendix 1.9
   (CutlerMerz repex results)". The independent repex review in Evoenergy's
   2024-29 proposal is **Appendix 2.3, by Qubist**, not CutlerMerz. I read it:
   it covers overhead distribution equipment, conductors, poles, transmission
   structures, protection, DC supply, SCADA, communications, zone substations,
   ground assets and metering. **It contains nothing on distribution
   transformers.** Chasing that reference will waste an afternoon.

2. **Do not confuse power transformers with distribution transformers.**
   Evoenergy's capex documents say "approximately 40 per cent of all Evoenergy
   power transformers are older than the stated service life", and cite a $3.4m
   transformer replacement at Telopea Park. Those are zone substation power
   transformers, not the LV distribution transformers in your fleet. The $3.4m
   figure in particular would be catastrophic to quote by mistake.

---

## 6. Standard asset life — PARTLY SOURCED

**Powercor/CitiPower state an expected service life of approximately 55 years**
for distribution transformers (PAL BUS 4.06, Jan 2025). Your
`standard_asset_life_years: 50` is therefore in a defensible range, and slightly
conservative.

Two useful fleet-age anchors: Ergon reports about **7,000 of 105,500**
distribution transformers over 50 years old; Powercor forecasts replacing about
1% of its population annually, which it notes "means our distribution
transformers on average would need to last 105 years before we replace them".
That gap between stated service life and observed replacement rate is worth a
sentence, because it is direct evidence that **transformers routinely outlive
their nominal life**, which is the same point your ageing model makes from the
thermal side.

### Still open: Evoenergy's own figure

Your yaml already makes the right distinction between the **regulatory
depreciation life** (drives the RAB and the tariff step) and the **physical
replacement age** (what your annuity needs). Keep it. For the regulatory one,
the source is:

- *AER Final Decision Attachment 4 — Regulatory depreciation — Evoenergy —
  2024-29*, and
- *AER Final decision — Evoenergy distribution determination 2024-29 —
  Depreciation module — Distribution* (an Excel model, which is where the
  standard lives by asset class actually live)

Both are on the AER's Evoenergy 2024-29 final decision page. I could not extract
the asset-life table from the Excel model remotely.

---

## What I could not get, and what you need to do

| # | Item | Why it is stuck | What to do |
|---|---|---|---|
| 1 | Evoenergy distribution transformer unit cost | Lives in Category Analysis RIN spreadsheets, not a fetchable PDF | Download Evoenergy's Category Analysis RIN response from the AER site |
| 2 | Evoenergy standard asset life | In the Depreciation module Excel | Download Attachment 4 and the Depreciation module from the Evoenergy 2024-29 final decision page |
| 3 | NGA 2025 scope 2 and scope 3 factors, confirmed | dcceew.gov.au blocks automated fetching | Download the NGA Factors 2025 PDF and read the scope 2 table |
| 4 | ACCU price, confirmed at source | CER site did not respond | Open the CER quarterly carbon market report directly |
| 5 | Marginal emissions intensity | Does not exist as a published Australian figure | Choose one of the three options in §2 and defend it |
| 6 | Time-weighted NSW spot over your simulated intervals | Needs NEM price data joined to your timestamps | Pull 30-minute NSW1 prices from AEMO for the two windows |

Item 1 is the one that moves the answer. Items 3 and 4 are confirmations of
figures I have already given you. Item 5 is a decision, not a lookup, and it
will not resolve itself by searching harder.

---

## Sources

- AER, *State of the energy market 2024*, Chapter 2 (National Electricity Market)
- AEMO, *Quarterly Energy Dynamics Q2 2026*
- DCCEEW, *Australian National Greenhouse Accounts Factors 2025* (via secondary compilation; confirm at source)
- NSW DCCEEW, *Greenhouse gas emissions accounting and reporting guidelines*, June 2025
- Clean Energy Regulator, *Quarterly Carbon Market Report*, Q1 2026 (via market aggregator; confirm at source)
- ACT Government, Climate Choices, *Solar feed-in tariff*
- Ergon Energy, *Business Case: Distribution Transformer Replacements*, January 2024
- Powercor/CitiPower, *Asset class overview: distribution transformers* (PAL BUS 4.06), January 2025
- Evoenergy, *Attachment 1: Capital expenditure*, January 2023
- Qubist, *Appendix 2.3: Independent Review Repex Portfolio* for Evoenergy, November 2023
