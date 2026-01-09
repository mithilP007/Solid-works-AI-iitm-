import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Video,
    VideoOff,
    Activity,
    Shield
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export default function LiveCamera() {
    const [cameraActive, setCameraActive] = useState(false);
    const [counts, setCounts] = useState({ bolt: 0, nut: 0, washer: 0, locatingpin: 0 });
    const [status, setStatus] = useState('Disconnected');

    useEffect(() => {
        let ws: WebSocket;
        if (cameraActive) {
            setStatus('Operational');
            ws = new WebSocket('ws://localhost:8000/ws');
            ws.onmessage = (event) => {
                const message = event.data;
                if (message.startsWith('LIVE_COUNTS:')) {
                    try {
                        const countsData = JSON.parse(message.substring(12));
                        setCounts(countsData);
                    } catch (err) {
                        console.error("Failed to parse live counts:", err);
                    }
                }
            };
            ws.onerror = (err) => {
                console.error("WebSocket error:", err);
                setStatus('Link Error');
            };
            ws.onclose = () => {
                if (cameraActive) setStatus('Reconnect...');
            };
        } else {
            setStatus('Disconnected');
        }
        return () => {
            if (ws) ws.close();
        };
    }, [cameraActive]);

    return (
        <div className="space-y-8 max-w-6xl mx-auto">
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-4xl font-black italic tracking-tighter text-white uppercase flex items-center gap-4">
                        <Video className="w-10 h-10 text-primary" />
                        Live Neural Perception
                    </h1>
                    <p className="text-muted-foreground mt-2 uppercase text-xs font-bold tracking-widest opacity-60">Real-time inference on edge assembly streams</p>
                </div>
                <div className="flex gap-4">
                    <Button
                        variant={cameraActive ? "destructive" : "default"}
                        size="lg"
                        className={cn("rounded-2xl h-14 px-8 font-black uppercase text-xs tracking-widest", !cameraActive && "bg-primary text-black hover:bg-primary/80")}
                        onClick={async () => {
                            if (cameraActive) {
                                await fetch('http://localhost:8000/camera/stop', { method: 'POST' });
                                setCameraActive(false);
                            } else {
                                setCameraActive(true);
                            }
                        }}
                    >
                        {cameraActive ? <><VideoOff className="w-5 h-5 mr-3" /> Terminate Stream</> : <><Video className="w-5 h-5 mr-3" /> Initialise Engine</>}
                    </Button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                <div className="lg:col-span-3">
                    <Card className="bg-sidebar/20 border-sidebar-border rounded-[2.5rem] overflow-hidden relative shadow-2xl shadow-primary/5">
                        <CardContent className="p-2">
                            {cameraActive ? (
                                <div className="relative aspect-video bg-black rounded-[2rem] overflow-hidden border border-white/5">
                                    <img
                                        src="http://localhost:8000/video_feed"
                                        className="w-full h-full object-contain"
                                        alt="Live perception stream"
                                    />
                                    <div className="absolute top-8 left-8 flex items-center gap-4">
                                        <div className="flex h-3 w-3 relative">
                                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                                            <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                                        </div>
                                        <span className="bg-black/80 backdrop-blur-md border border-white/10 px-4 py-2 rounded-full text-[10px] font-black text-white uppercase tracking-[0.3em]">Neural Link: Active</span>
                                    </div>
                                </div>
                            ) : (
                                <div className="aspect-video bg-sidebar-accent/30 rounded-[2rem] flex flex-col items-center justify-center border-2 border-dashed border-sidebar-border opacity-50 group transition-all hover:opacity-100 cursor-pointer" onClick={() => setCameraActive(true)}>
                                    <Video className="w-20 h-20 text-muted-foreground mb-6 opacity-20 group-hover:scale-110 transition-transform duration-500" />
                                    <p className="text-sm font-black text-muted-foreground uppercase tracking-widest">Awaiting Engine Signal</p>
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </div>

                <div className="space-y-6">
                    <div className="glass-card p-6 bg-primary/5 border-primary/20">
                        <div className="flex items-center gap-3 mb-4">
                            <Shield className="w-5 h-5 text-primary" />
                            <span className="text-xs font-black uppercase text-white tracking-widest">Engine Status</span>
                        </div>
                        <p className="text-2xl font-black italic uppercase text-primary mb-1">{status}</p>
                        <p className="text-[10px] text-muted-foreground uppercase font-bold">Latency: {cameraActive ? '2ms' : 'N/A'}</p>
                    </div>

                    <div className="space-y-3">
                        <h3 className="text-xs font-black text-muted-foreground uppercase tracking-widest px-2">Perception Buffer</h3>
                        {Object.entries(counts).map(([cls, count]) => (
                            <div key={cls} className="glass-card p-5 group hover:border-primary/30 transition-all">
                                <div className="flex justify-between items-center">
                                    <span className="text-[10px] font-black uppercase text-muted-foreground group-hover:text-white transition-colors tracking-widest">{cls}</span>
                                    <span className={cn("text-2xl font-black transition-all", count > 0 ? "text-primary" : "text-muted-foreground/20")}>{count}</span>
                                </div>
                                {count > 0 && <div className="mt-2 h-1 w-full bg-primary/10 rounded-full overflow-hidden"><motion.div initial={{ width: 0 }} animate={{ width: '100%' }} className="h-full bg-primary" /></div>}
                            </div>
                        ))}
                    </div>

                    <div className="p-6 rounded-3xl bg-sidebar-accent/50 border border-sidebar-border">
                        <div className="flex items-center gap-2 mb-4">
                            <Activity className="w-4 h-4 text-accent" />
                            <span className="text-[10px] font-black uppercase text-accent tracking-widest">Edge Analysis</span>
                        </div>
                        <p className="text-[10px] leading-relaxed text-muted-foreground font-medium italic uppercase tracking-tighter">
                            Optimized High-Fidelity Mode (1024px). Balanced for maximum small-part detection accuracy and fluid real-time tracking.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
