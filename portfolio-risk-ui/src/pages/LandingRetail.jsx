import {
  LandingPage, LandingHeader, LandingHero, LandingFeatureGrid, LandingHowItWorks, LandingFooter,
} from '../components/LandingLayout'

const FEATURES = [
  {
    title: 'A thousand possible futures',
    body: "We simulate a thousand different ways the next year could go for your money, so you see a range of outcomes, not one guess.",
  },
  {
    title: 'Survive 2008 or COVID?',
    body: "See exactly how your portfolio would have performed during the world's worst market crashes.",
  },
  {
    title: 'Are your eggs really in different baskets?',
    body: "Most people think they're diversified. We show you if your stocks actually move together, meaning they're not.",
  },
  {
    title: 'Getting paid enough for your risk?',
    body: 'Some portfolios take big risks for small rewards. We show you if yours is one of them.',
  },
  {
    title: "What if I'd invested this way before?",
    body: 'Test your strategy against real history before trusting it with real money.',
  },
  {
    title: 'The best possible mix',
    body: 'See if a different split of your same stocks could get you more return for the same risk.',
  },
]

const STEPS = [
  'Tell us what you own',
  'We run the numbers',
  'Understand your risk in plain English',
]

export default function LandingRetail() {
  return (
    <LandingPage>
      <LandingHeader crossLinkLabel="For professionals →" crossLinkTo="/pro" />
      <LandingHero
        headline="See what could happen to your money before it happens"
        subhead="Varense runs your portfolio through thousands of simulated futures, including real crashes like 2008 and 2020, so you know what you're actually risking, not just what you're earning."
      />
      <LandingFeatureGrid features={FEATURES} />
      <LandingHowItWorks steps={STEPS} />
      <LandingFooter />
    </LandingPage>
  )
}
