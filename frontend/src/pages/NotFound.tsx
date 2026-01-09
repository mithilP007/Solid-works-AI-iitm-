import { useNavigate } from 'react-router-dom';
import { Home } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function NotFound() {
    const navigate = useNavigate();

    return (
        <div className="h-[70vh] flex flex-col items-center justify-center text-center space-y-8">
            <div className="relative">
                <h1 className="text-[15rem] font-black text-white/5 uppercase italic tracking-tighter leading-none">404</h1>
                <div className="absolute inset-0 flex items-center justify-center">
                    <p className="text-3xl font-black italic uppercase text-primary glow-text-cyan">Coordinates Lost</p>
                </div>
            </div>
            <p className="text-muted-foreground max-w-md mx-auto uppercase text-xs font-bold tracking-[0.3em]">
                The requested neural sector is currently outside of perception coverage.
            </p>
            <Button
                size="lg"
                className="bg-primary text-black font-black uppercase h-14 px-10 rounded-2xl shadow-xl shadow-primary/20"
                onClick={() => navigate('/')}
            >
                <Home className="w-5 h-5 mr-3" />
                Return to Dashboard
            </Button>
        </div>
    );
}
