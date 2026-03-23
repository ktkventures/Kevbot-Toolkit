'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

const quizQuestions = [
  {
    question: 'How do you feel about trading decisions?',
    options: [
      { text: 'I want someone else to handle it', trait: 'subscriber' },
      { text: 'I like analyzing charts and data', trait: 'strategy' },
      { text: 'I want to manage a diversified portfolio', trait: 'portfolio' },
      { text: 'I build the tools others use', trait: 'pack' },
    ],
  },
  {
    question: 'What\'s your ideal time commitment?',
    options: [
      { text: 'Set it and forget it', trait: 'subscriber' },
      { text: 'A few hours a week testing ideas', trait: 'strategy' },
      { text: 'I\'m always optimizing', trait: 'portfolio' },
      { text: 'I live and breathe indicators', trait: 'pack' },
    ],
  },
  {
    question: 'What matters most to you?',
    options: [
      { text: 'Passive income with minimal effort', trait: 'subscriber' },
      { text: 'Having edge through data-driven strategies', trait: 'strategy' },
      { text: 'Building something others pay for', trait: 'portfolio' },
      { text: 'Pushing the boundaries of technical analysis', trait: 'pack' },
    ],
  },
];

const pathDetails: Record<string, { title: string; description: string; features: string[]; price: string; color: string }> = {
  subscriber: {
    title: 'The Copy Trader',
    description: 'You prefer a hands-off approach. Subscribe to proven portfolios and let the alerts work for you.',
    features: ['Browse marketplace', 'Free portfolio rotation', 'Webhook alerts', 'Performance tracking'],
    price: 'Free',
    color: 'var(--green)',
  },
  strategy: {
    title: 'The Strategy Builder',
    description: 'You love the data. Build, backtest, and forward-test your own trading strategies.',
    features: ['Strategy Builder', 'Mass Backtesting', 'Forward testing', 'Publish to marketplace'],
    price: '$29/mo',
    color: 'var(--blue)',
  },
  portfolio: {
    title: 'The Portfolio Architect',
    description: 'You see the big picture. Curate strategies into portfolios and earn revenue from subscribers.',
    features: ['Portfolio construction', 'Prop firm compliance', 'Revenue sharing', 'Unlimited subscriptions'],
    price: '$49/mo',
    color: 'var(--accent)',
  },
  pack: {
    title: 'The Tool Builder',
    description: 'You push boundaries. Develop custom indicators, interpreters, and triggers for the community.',
    features: ['Code editor', 'Indicator development', 'Pack royalties', 'API access'],
    price: '$79/mo',
    color: 'var(--purple)',
  },
};

const roleCards = [
  { id: 'subscriber', label: 'Copy Trade', sublabel: 'Free', color: 'var(--green)' },
  { id: 'strategy', label: 'Build Strategies', sublabel: '$29/mo', color: 'var(--blue)' },
  { id: 'portfolio', label: 'Build Portfolios', sublabel: '$49/mo', color: 'var(--accent)' },
  { id: 'pack', label: 'Create Packs', sublabel: '$79/mo', color: 'var(--purple)' },
];

export default function OnboardingV4() {
  const [phase, setPhase] = useState<'quiz' | 'result' | 'select' | 'celebrate'>('quiz');
  const [quizStep, setQuizStep] = useState(0);
  const [answers, setAnswers] = useState<string[]>([]);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [hoveredOption, setHoveredOption] = useState<number | null>(null);

  function handleAnswer(trait: string) {
    const newAnswers = [...answers, trait];
    setAnswers(newAnswers);

    if (quizStep < quizQuestions.length - 1) {
      setQuizStep(quizStep + 1);
    } else {
      // Tally results
      const counts: Record<string, number> = {};
      newAnswers.forEach((a) => { counts[a] = (counts[a] || 0) + 1; });
      const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
      setSelectedPath(top);
      setPhase('result');
    }
  }

  function resetQuiz() {
    setQuizStep(0);
    setAnswers([]);
    setSelectedPath(null);
    setPhase('quiz');
  }

  // Quiz Phase
  if (phase === 'quiz') {
    const q = quizQuestions[quizStep];
    return (
      <div className="max-w-lg mx-auto">
        <PageHeader
          title="What kind of trader are you?"
          subtitle={`Question ${quizStep + 1} of ${quizQuestions.length}`}
        />

        {/* Progress dots */}
        <div className="flex gap-2 justify-center mb-8">
          {quizQuestions.map((_, i) => (
            <div
              key={i}
              className="w-3 h-3 rounded-full transition-all duration-300"
              style={{
                background: i < quizStep ? 'var(--accent)' : i === quizStep ? 'var(--accent)' : 'var(--bg-input)',
                transform: i === quizStep ? 'scale(1.3)' : 'scale(1)',
              }}
            />
          ))}
        </div>

        <h3 className="text-lg font-semibold text-center mb-6">{q.question}</h3>

        <div className="space-y-3">
          {q.options.map((opt, i) => (
            <button
              key={i}
              onClick={() => handleAnswer(opt.trait)}
              onMouseEnter={() => setHoveredOption(i)}
              onMouseLeave={() => setHoveredOption(null)}
              className="w-full p-4 rounded-xl text-left transition-all duration-200"
              style={{
                background: hoveredOption === i ? 'var(--accent-muted)' : 'var(--bg-card)',
                border: `2px solid ${hoveredOption === i ? 'var(--accent)' : 'var(--border)'}`,
                transform: hoveredOption === i ? 'translateX(4px)' : 'translateX(0)',
              }}
            >
              <p className="text-sm font-medium">{opt.text}</p>
            </button>
          ))}
        </div>
      </div>
    );
  }

  // Result Phase
  if (phase === 'result' && selectedPath) {
    const detail = pathDetails[selectedPath];
    return (
      <div className="max-w-lg mx-auto text-center">
        <div className="mb-8">
          <div
            className="w-20 h-20 rounded-full flex items-center justify-center text-3xl mx-auto mb-4"
            style={{
              background: detail.color + '22',
              border: `3px solid ${detail.color}`,
              boxShadow: `0 0 30px ${detail.color}44`,
            }}
          >
            {selectedPath === 'subscriber' ? '$' : selectedPath === 'strategy' ? '\u{1F4CA}' : selectedPath === 'portfolio' ? '\u{1F4BC}' : '\u{1F9E9}'}
          </div>
          <h2 className="text-2xl font-bold mb-1">You&apos;re {detail.title}!</h2>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{detail.description}</p>
        </div>

        <Card className="mb-6 text-left">
          <h4 className="text-sm font-semibold mb-3">What you&apos;ll get:</h4>
          <ul className="space-y-2">
            {detail.features.map((f) => (
              <li key={f} className="flex items-center gap-2 text-sm">
                <span style={{ color: detail.color }}>&#10003;</span>
                <span style={{ color: 'var(--text-secondary)' }}>{f}</span>
              </li>
            ))}
          </ul>
          <div className="mt-4 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
            <span className="text-lg font-bold" style={{ color: detail.color }}>{detail.price}</span>
          </div>
        </Card>

        <div className="flex gap-3">
          <button
            onClick={() => setPhase('celebrate')}
            className="flex-1 py-3 rounded-lg text-sm font-medium"
            style={{ background: detail.color, color: 'var(--bg-primary)' }}
          >
            This is me!
          </button>
          <button
            onClick={() => setPhase('select')}
            className="px-4 py-3 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
          >
            Choose manually
          </button>
        </div>

        <button
          onClick={resetQuiz}
          className="mt-3 text-xs"
          style={{ color: 'var(--text-muted)' }}
        >
          Retake quiz
        </button>
      </div>
    );
  }

  // Manual Select Phase
  if (phase === 'select') {
    return (
      <div className="max-w-lg mx-auto">
        <PageHeader
          title="Choose your path"
          subtitle="Select the option that fits you best"
        />
        <div className="space-y-3 mb-6">
          {roleCards.map((role) => (
            <button
              key={role.id}
              onClick={() => { setSelectedPath(role.id); setPhase('celebrate'); }}
              className="w-full p-4 rounded-xl text-left transition-all duration-200 flex items-center gap-4"
              style={{
                background: 'var(--bg-card)',
                border: `2px solid var(--border)`,
              }}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = role.color)}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--border)')}
            >
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center text-lg shrink-0"
                style={{ background: role.color + '22', color: role.color }}
              >
                {role.id === 'subscriber' ? '$' : role.id === 'strategy' ? '\u{1F4CA}' : role.id === 'portfolio' ? '\u{1F4BC}' : '\u{1F9E9}'}
              </div>
              <div className="flex-1">
                <p className="font-semibold">{role.label}</p>
                <p className="text-xs" style={{ color: role.color }}>{role.sublabel}</p>
              </div>
              <span style={{ color: 'var(--text-muted)' }}>&rarr;</span>
            </button>
          ))}
        </div>

        <button
          onClick={() => setPhase('result')}
          className="text-sm"
          style={{ color: 'var(--text-muted)' }}
        >
          &larr; Back to recommendation
        </button>
      </div>
    );
  }

  // Celebrate Phase
  return (
    <div className="max-w-lg mx-auto text-center py-12">
      <div className="relative inline-block mb-6">
        <div
          className="w-24 h-24 rounded-full flex items-center justify-center text-5xl mx-auto"
          style={{
            background: 'var(--green-muted)',
            color: 'var(--green)',
            boxShadow: '0 0 40px var(--green-muted)',
          }}
        >
          &#10003;
        </div>
        {/* Confetti dots */}
        {[...Array(8)].map((_, i) => {
          const angle = (i / 8) * Math.PI * 2;
          const x = Math.cos(angle) * 60;
          const y = Math.sin(angle) * 60;
          const colors = ['var(--accent)', 'var(--green)', 'var(--blue)', 'var(--orange)', 'var(--purple)', 'var(--teal)', 'var(--red)', 'var(--accent)'];
          return (
            <div
              key={i}
              className="absolute w-2 h-2 rounded-full"
              style={{
                background: colors[i],
                left: `calc(50% + ${x}px - 4px)`,
                top: `calc(50% + ${y}px - 4px)`,
              }}
            />
          );
        })}
      </div>

      <h2 className="text-2xl font-bold mb-2">Welcome aboard!</h2>
      <p className="text-sm mb-2" style={{ color: 'var(--text-muted)' }}>
        Your account is ready. Let&apos;s get started.
      </p>
      {selectedPath && (
        <p className="text-sm mb-8">
          <span style={{ color: pathDetails[selectedPath].color }}>
            {pathDetails[selectedPath].title}
          </span>
          {' '}&middot;{' '}
          <span style={{ color: 'var(--text-muted)' }}>{pathDetails[selectedPath].price}</span>
        </p>
      )}

      <div className="space-y-3 max-w-sm mx-auto mb-8">
        {[
          { text: 'Explore the marketplace', done: false },
          { text: 'Set up your webhook', done: false },
          { text: 'Configure risk settings', done: false },
        ].map((item, i) => (
          <div
            key={i}
            className="flex items-center gap-3 p-3 rounded-lg text-left"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
          >
            <div
              className="w-6 h-6 rounded-full flex items-center justify-center text-xs shrink-0"
              style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
            >
              {i + 1}
            </div>
            <span className="text-sm">{item.text}</span>
          </div>
        ))}
      </div>

      <button
        className="px-8 py-3 rounded-lg text-sm font-medium"
        style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
      >
        Go to Dashboard
      </button>
    </div>
  );
}
