# RoR Trader — Monetization Model & Platform Vision

## Mission Statement
Make jobs optional. Provide anyone — regardless of wealth, location, or technical skill — with access to data-driven passive income through systematized trading strategies.

## Core Philosophy
- **Accessible:** Free tier gets real value, not just a teaser
- **Meritocratic:** Best strategies surface naturally through performance data
- **Ecosystem-driven:** Contributors at every level are compensated
- **Legal-first:** Signal service model (users trade their own accounts), not investment management

---

## Platform Architecture

### Value Chain (Bottom-Up)
```
Pack Creators      → Build technical indicators, interpreters, triggers
Strategy Builders  → Use packs to create and test trading strategies
Portfolio Builders → Curate strategies into diversified portfolios
Subscribers        → Subscribe to portfolios, receive alerts, trade
```

Each layer adds value. Revenue flows down from subscribers and is shared across the chain.

### User Roles & Tiers

#### 1. Subscriber (Free)
- Browse marketplace
- Access curated free portfolio rotation (selected by admin, rotates periodically)
- Subscribe to free portfolios
- Configure webhook URL and risk per trade
- View portfolio performance and alert history
- Locking in a free-rotation portfolio keeps access even after rotation ends

#### 2. Subscriber (Premium) — $X/month per portfolio
- Subscribe to any marketplace portfolio
- Choose from hundreds of community-contributed portfolios
- Full alert history, trade log, and performance analytics
- Priority webhook delivery
- Portfolio pricing set by creator (with platform-enforced floor based on verification level)

#### 3. Strategy Builder — $X/month subscription
- Access to Strategy Builder, Mass Builder, backtesting engine
- License/subscribe to confluence packs from marketplace
- Create, test, and forward-test strategies
- Publish strategies to marketplace for portfolio builders to use
- Earn revenue share when strategies are used in sold portfolios

#### 4. Portfolio Builder — $X/month subscription (includes Strategy Builder access)
- Assemble portfolios from available strategies
- Run portfolio-level analytics (correlation, drawdown, prop firm compliance)
- Publish portfolios to marketplace
- Set pricing for portfolio subscriptions
- Earn revenue from subscribers (minus platform fee and strategy/pack royalties)

#### 5. Pack Creator — $X/month subscription (includes all access)
- Build custom confluence packs (indicators, interpreters, triggers)
- Publish packs to marketplace
- Earn royalties when packs are used in sold portfolios
- Access to indicator development tools, code editor, testing framework

#### 6. Admin
- Platform management, content moderation, user management
- Curate free-rotation portfolios
- Monitor platform health and revenue

---

## Revenue Streams

### 1. Platform Fee on Marketplace Transactions (Primary)
- 15-20% of all marketplace revenue (portfolio subscriptions, strategy licensing, pack licensing)
- Similar to Fiverr (20%), Amazon (15%), Etsy (6.5% + fees)
- Revenue flows: Subscriber pays → Platform takes cut → Remainder split among creator chain

### 2. Creator Subscription Tiers (Recurring)
- Monthly fee for Strategy Builder, Portfolio Builder, Pack Creator roles
- Provides access to creation/testing tools
- Pricing TBD but should be accessible (target: $20-50/month range)

### 3. Portfolio Subscriptions (Recurring)
- Monthly fee set by portfolio creator (with platform floor)
- Pricing factors: verification level, alert count, forward-test duration, live performance track record
- Suggested tiers:
  - Unverified (backtest only): $5-15/month
  - Forward-tested (3+ months): $15-50/month
  - Live-verified (6+ months, 500+ alerts): $50-200/month

### 4. Prop Firm Affiliate Commissions
- Referral links to compatible prop firms (TTP, FTMO, Topstep, etc.)
- Portfolios show which prop firm evaluations they'd pass
- Commission per sign-up (typically $50-150 per referral)
- Natural fit: users need funded accounts, we prove the strategies work on those accounts

### 5. Revenue Sharing Model (Creator Chain)
When a subscriber pays for a portfolio:
```
$30/month portfolio subscription
  → Platform fee: 15% = $4.50
  → Portfolio creator: 50% of remainder = $12.75
  → Strategy creators: 25% of remainder (split among strategies used) = $6.375
  → Pack creators: 10% of remainder (split among packs used) = $2.55
  → Reserve/community fund: remainder
```
Exact splits TBD. Key principle: every contributor gets paid.

---

## Marketplace Design

### Portfolio Marketplace (Primary)
- Search/filter by: asset type, return, risk level, prop firm compatibility, price, subscriber count, verification level
- Verification badges:
  - Backtest Only (gray)
  - Forward Tested (blue) — 3+ months maintaining performance
  - Live Verified (gold) — 6+ months, 500+ real alerts, performance maintained
- Performance metrics: Win Rate, PF, Daily R, Max DD, Sharpe, alert count, subscriber count
- Reviews/ratings from subscribers
- Free rotation section (curated by admin)

### Strategy Marketplace
- Available to Portfolio Builders
- Browse/license individual strategies
- Performance data, confluence requirements, execution details
- Pricing: subscription or revenue-share with portfolio sales

### Pack Marketplace
- Available to Strategy Builders
- Browse/license confluence packs
- Indicator descriptions, output states, trigger definitions
- Pricing: subscription or revenue-share

---

## Legal Framework

### Signal Service Model (NOT Investment Management)
Critical distinctions that keep us outside investment advisor registration:
1. **Users trade their own accounts** — we never have access to their money
2. **Users configure their own webhooks** — they choose how to connect
3. **Users set their own risk** — we suggest, they decide
4. **We provide tools and information** — not personalized investment advice
5. **Performance is historical** — past results don't guarantee future performance

### Required Disclaimers
- "RoR Trader is an educational and analytical platform. It does not provide personalized investment advice."
- "Past performance does not guarantee future results. Trading involves risk of loss."
- "Users are responsible for their own trading decisions."
- "RoR Trader does not manage funds or have access to user brokerage accounts."

### Recommended Legal Steps
- [ ] Consult securities attorney before paid launch
- [ ] Draft Terms of Service with signal service language
- [ ] Draft Privacy Policy (user data, webhook URLs, trading data)
- [ ] Establish entity structure (LLC at minimum)
- [ ] Review prop firm affiliate program terms
- [ ] Investigate state-by-state signal service regulations

---

## Strategic Rollout Phases

### Phase 1: Personal Use & Validation (Current)
- Build and test strategies personally
- Prove the system works with real trading data
- Refine the platform (current frontend work)
- No external users

### Phase 2: Closed Beta (Next)
- Invite 10-20 trusted users
- Free access to test the subscriber experience
- Validate webhook delivery, alert accuracy, onboarding flow
- Collect feedback on UX and portfolio subscription experience

### Phase 3: Curated Launch
- Open to public with free tier (curated portfolios only)
- No marketplace yet — admin-curated portfolios
- AI agents (Claude) can help create/maintain strategies and portfolios
- Prop firm affiliate revenue starts
- Monthly creator subscription available for power users

### Phase 4: Marketplace Launch
- Open marketplace for portfolios, strategies, packs
- Revenue sharing model active
- Verification system live (backtest → forward test → live verified)
- Creator payouts operational

### Phase 5: Scale
- International expansion
- Additional asset types (crypto, forex, futures)
- Mobile app
- API for institutional users
- Community fund / nonprofit arm for displaced workers

---

## Close-to-Chest vs. Open Ecosystem

### Option A: Close-to-Chest (Recommended for Phase 1-3)
- AI agents create and maintain strategies/portfolios
- Users only see the subscriber experience
- Protects IP during early vulnerable stage
- Simpler to manage, fewer edge cases
- Can still build the full UI (marketplace, creator tools) for when it's ready

### Option B: Open Ecosystem (Recommended for Phase 4+)
- Community contributes at all levels
- Marketplace drives growth through network effects
- More diverse strategies from many perspectives
- Revenue sharing incentivizes quality
- By this point, the platform has enough moat (data, track record, brand) to withstand copycats

### Hybrid Approach (Recommended)
Start with A, transition to B. Build the UI for B now (it serves as the admin interface in Phase A anyway), but don't expose creator roles to public until Phase 4.

---

## Humanitarian Component

### Vision
- Portion of platform revenue funds initiatives for people displaced by AI/automation
- Partner with nonprofits, churches, state agencies
- Provide free portfolio access to verified displaced workers
- Build financial literacy education into the platform

### Potential Structures
- Community fund (% of platform fee)
- Nonprofit arm for education and workforce transition
- Scholarship program for creator tier access
- Partnership with workforce development organizations

---

## UI Pages Required

### New Pages for Monetization Support
1. **Marketplace** — Browse portfolios, strategies, packs with search/filter/sort
2. **Portfolio Subscription Flow** — Subscribe, configure webhooks, set risk, billing
3. **Creator Dashboard** — Earnings, subscribers, performance, payouts
4. **Pricing / Plans** — Tier comparison, upgrade flow
5. **Onboarding Wizard** — Guided setup for new users
6. **Webhook Setup Wizard** — Simplified broker connection
7. **Prop Firm Hub** — Compatible firms, affiliate links, evaluation tracking
8. **User Profile & Roles** — Account management, tier display, achievements
9. **Admin Dashboard** — Platform metrics, user management, content moderation, curation
10. **Earnings & Payouts** — For creators: revenue tracking, payout history, tax reporting

### Modifications to Existing Pages
- **Portfolios** — Add "Publish to Marketplace" flow
- **Strategies** — Add "List on Marketplace" option
- **Confluence Packs** — Add "Publish Pack" option
- **Settings/Account** — Add subscription management, billing, payout config
