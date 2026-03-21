import Card from '@/components/Card';
import ThemeSwitcher from '@/components/ThemeSwitcher';

export default function ThemesPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Themes</h1>
      <div className="max-w-3xl">
        <Card>
          <ThemeSwitcher />
        </Card>
      </div>
    </div>
  );
}
