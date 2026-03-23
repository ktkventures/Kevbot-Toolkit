'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-path-cards',
      name: 'Path Cards',
      description: 'Welcome message with 2 path cards: "I want to trade" vs "I want to create".',
      rationale: 'Simplest possible onboarding. Two big clickable cards — one for subscribers (free, hands-off) and one for creators (paid, hands-on). Selection highlights the card and enables the Continue button. No wizard, no steps.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-wizard',
      name: 'Full Wizard',
      description: '5-step onboarding: role selection (4 paths), profile setup (name, avatar, experience), broker connection (webhook test), first action (context-aware), success screen with next steps.',
      rationale: 'Complete guided setup. Step 1 presents 4 role cards with pricing. Step 2 collects display name, avatar placeholder, and experience level. Step 3 handles broker selection and webhook URL with test button (skippable). Step 4 offers context-aware first action based on chosen role. Step 5 celebrates completion with ordered next steps. Progress bar tracks position through all 5 steps.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: '3 screens: choose path, set up webhook, done.',
      rationale: 'Fastest path to value. Screen 1: two simple button choices (Subscribe & Trade or Create & Earn). Screen 2: webhook URL input with skip option. Screen 3: success with context-aware CTA. No profile setup, no experience level — those can be filled in later from settings.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Personality quiz ("What kind of trader are you?"), animated result reveal, manual override option, celebration screen with confetti dots.',
      rationale: 'Makes onboarding fun. 3-question personality quiz with hover animations recommends a path (The Copy Trader, The Strategy Builder, The Portfolio Architect, or The Tool Builder). Result screen shows the recommendation with features and pricing. Users can accept or choose manually. Celebration screen has decorative confetti dots and ordered next steps. Quiz is retakeable.',
    },
    component: V4,
  },
];

export default function OnboardingPage() {
  return <VersionedPage pageKey="onboarding" versions={versions} />;
}
