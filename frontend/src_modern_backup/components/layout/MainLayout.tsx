import { Sidebar } from './Sidebar';

export function MainLayout({ children }: { children: React.ReactNode }) {
    return (
        <div className="flex min-h-screen bg-background font-sans selection:bg-primary selection:text-black">
            <Sidebar />
            <main className="flex-1 overflow-y-auto px-8 py-12">
                {children}
            </main>
        </div>
    );
}
