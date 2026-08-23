# Backlog and Next Steps

This page records intended work, not shipped functionality. Items move to
[Current State](Current-State.md) only after implementation and verification.

## Next initiative: universal integer-pool lottery benchmark

Evolve LottoBench from a EuroMillions-centred history shape into a universal benchmark for lottery
draws composed of integer-valued pools. A game may have one main pool and zero or more auxiliary
pools, such as stars, bonus balls, supplementary numbers, or a separately ranged special number.

The generic core must not hard-code the words `star` or `EuroMillions`. Those remain valid names in
individual game definitions and compatibility interfaces.

### Proposed work

1. **Universal game and draw schema**
   - Define each pool by name, inclusive integer range, draw count, uniqueness, and whether order
     matters.
   - Represent a draw as a mapping of pool name to integer values.
   - Store jurisdiction, game identifier, draw time, source provenance, and retrieval time.

2. **Versioned rule epochs and exclusions**
   - Give every rule configuration an effective start and optional end date.
   - Prevent incompatible epochs from being combined silently.
   - Record excluded periods with a reason, rather than deleting observations.
   - Make epoch selection a core benchmark option instead of a provider-specific workaround.

3. **Canonical storage and ingestion**
   - Use the multi-game SQLite store as the canonical local representation.
   - Keep CSV as an import/export format.
   - Add source adapters that preserve raw provenance, checksums, corrections, and field mappings.
   - Reconcile conflicting sources explicitly; automated search may discover candidates but must not
     silently promote unverified data.

4. **Provider migration**
   - Change maintained providers to consume the universal `GameSpec` and pool-based history.
   - Require every provider to declare supported pool shapes and optional dependencies.
   - Preserve the current EuroMillions-facing API through a tested compatibility layer.

5. **Time-safe contextual data**
   - Distinguish prediction generation, optional user submission, sales cutoff, draw execution,
     result publication, and ingestion timestamps.
   - Allow a run to use only features whose `available_at` time precedes its prediction cutoff.
   - Keep personal, payment, receipt, and wager-execution data outside LottoBench.

6. **Universal evaluation**
   - Validate every proposed ticket against every pool in the applicable rule epoch.
   - Report pool-level matches, combined matches, coverage, diversity, runtime, and deterministic
     reproducibility.
   - Compare realized ROI only at equal budget and retain raw stake and payout alongside normalized
     lift against the `uniform_random` control.
   - Keep anomaly-detection evidence separate from claims about prediction or realized ROI.

### Acceptance criteria

- A one-pool game and a main-plus-auxiliary-pool game run through the same public benchmark API.
- EuroMillions, UK Lotto, German LOTTO 6aus49, Danske Lotto, Netherlands Lotto, and Sweden Lotto are
  expressed as data-driven game/rule definitions rather than game-specific benchmark code.
- A rule change creates a new epoch and cannot leak incompatible historical rows into a run unless
  the user selects an explicit cross-epoch policy.
- At least one maintained provider, `uniform_random`, and realized-ROI settlement work end to end
  across all supported pool shapes.
- The same seed, data snapshot, rule epoch, provider version, and configuration produce identical
  output.
- Existing EuroMillions public entry points either remain compatible or emit a documented migration
  warning during the alpha series.

### Explicitly deferred

- Automatic wager or ticket submission.
- Claims that contextual or inferred latent variables predict a correctly operated random draw.
- Central collection of identifying, payment, or receipt data.
- Unreviewed web-search results entering the canonical benchmark dataset.

