import type { IconName } from '@/lib/catalog';

const P: Record<IconName | 'play' | 'pause' | 'prev' | 'next' | 'logo' | 'layers' | 'pin' | 'map', JSX.Element> = {
  temp: <path d="M14 4v10.5a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0z" />,
  wind: (
    <>
      <path d="M3 8h11a3 3 0 1 0-3-3" />
      <path d="M3 12h16a3 3 0 1 1-3 3" />
      <path d="M3 16h7" />
    </>
  ),
  gust: (
    <>
      <path d="M3 7h9a2.5 2.5 0 1 0-2.5-2.5" />
      <path d="M3 12h15a3 3 0 1 1-3 3" />
      <path d="M3 17h6a2 2 0 1 1-2 2" />
    </>
  ),
  drop: <path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z" />,
  cloud: <path d="M7 18h10a4 4 0 0 0 0-8 6 6 0 0 0-11.5 1.5A3.5 3.5 0 0 0 7 18z" />,
  humidity: (
    <>
      <path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z" />
      <path d="M9.5 16.5l5-5" />
      <circle cx="10" cy="12" r=".6" />
      <circle cx="14" cy="16" r=".6" />
    </>
  ),
  gauge: (
    <>
      <circle cx="12" cy="13" r="8" />
      <path d="M12 13l4-4" />
      <path d="M12 5V3" />
    </>
  ),
  bolt: <path d="M13 2L4 14h7l-1 8 9-12h-7z" />,
  play: <path d="M8 5v14l11-7z" fill="currentColor" stroke="none" />,
  pause: (
    <>
      <rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none" />
      <rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none" />
    </>
  ),
  prev: <path d="M15 6l-6 6 6 6" />,
  next: <path d="M9 6l6 6-6 6" />,
  logo: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M3 12c3-3 6-3 9 0s6 3 9 0" />
      <path d="M5 7c2.5-2 5-2 7 0s4.5 2 7 0" />
    </>
  ),
  layers: (
    <>
      <path d="M12 3l9 5-9 5-9-5z" />
      <path d="M3 13l9 5 9-5" />
    </>
  ),
  pin: (
    <>
      <path d="M12 21s7-6.2 7-12a7 7 0 0 0-14 0c0 5.8 7 12 7 12z" />
      <circle cx="12" cy="9" r="2.5" />
    </>
  ),
  map: (
    <>
      <path d="M9 4L3 6v14l6-2 6 2 6-2V4l-6 2z" />
      <path d="M9 4v14M15 6v14" />
    </>
  ),
};

export function Icon({ name, size = 20 }: { name: keyof typeof P; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {P[name]}
    </svg>
  );
}
