import { motion } from 'framer-motion';
import { Download, FileText, FileCode, ExternalLink, ShieldCheck, Database, Rocket } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function Export() {
    const downloadFile = async (format: string) => {
        const endpoint = format === 'csv' ? '/download' : `/download/${format}`;
        window.open(`http://localhost:8000${endpoint}`, '_blank');
    };

    return (
        <div className="max-w-6xl mx-auto space-y-12">
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-4xl font-black italic tracking-tighter text-white uppercase flex items-center gap-4">
                        <Download className="w-10 h-10 text-primary" />
                        Strategic Export Hub
                    </h1>
                    <p className="text-muted-foreground mt-2 uppercase text-xs font-bold tracking-widest opacity-60">Generate formal documentation for industrial workflows</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {[
                    { icon: FileCode, title: 'CSV Dataset', desc: 'Raw perception logs including part counts and confidence scores for batch datasets.', format: 'csv', color: 'primary' },
                    { icon: FileText, title: 'PDF Research', desc: 'Formalized detection summary including timestamps and aggregate industrial metrics.', format: 'pdf', color: 'accent' },
                    { icon: Rocket, title: 'DOCX Summary', desc: 'Technical summary report optimized for industrial engineering documentation.', format: 'docs', color: 'primary' }
                ].map((item, i) => (
                    <motion.div
                        key={item.format}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }}
                    >
                        <Card className="glass-card p-4 group hover:border-primary/30 transition-all border-sidebar-border bg-sidebar/20 h-full flex flex-col">
                            <CardHeader>
                                <div className="w-14 h-14 rounded-2xl bg-sidebar-accent flex items-center justify-center mb-4 group-hover:scale-110 transition-transform text-primary">
                                    <item.icon className="w-8 h-8" />
                                </div>
                                <CardTitle className="text-xl font-black italic uppercase tracking-tight text-white">{item.title}</CardTitle>
                            </CardHeader>
                            <CardContent className="flex-1">
                                <p className="text-sm text-muted-foreground mb-8 leading-relaxed italic">{item.desc}</p>
                                <Button
                                    className="w-full h-14 rounded-xl bg-sidebar-accent hover:bg-primary hover:text-black font-black uppercase tracking-widest transition-all"
                                    onClick={() => downloadFile(item.format)}
                                >
                                    GENERATE {item.format}
                                </Button>
                            </CardContent>
                        </Card>
                    </motion.div>
                ))}
            </div>

            <div className="glass-card p-12 bg-gradient-to-r from-primary/5 to-transparent border-primary/20 rounded-[3rem] relative overflow-hidden">
                <div className="absolute top-0 right-0 p-12 opacity-5 pointer-events-none">
                    <ShieldCheck className="w-64 h-64" />
                </div>
                <div className="max-w-2xl">
                    <h3 className="text-2xl font-black text-white italic uppercase mb-4 tracking-tighter">Verified Industrial Integrity</h3>
                    <p className="text-muted-foreground text-sm leading-relaxed mb-8">
                        Our export protocols ensure that all perception data is synchronized with the latest neural engine logs, providing a single source of truth for factory automation workflows.
                    </p>
                    <div className="flex gap-8 group cursor-pointer" onClick={() => window.open('http://localhost:8000/stats', '_blank')}>
                        <div className="flex items-center gap-2">
                            <Database className="w-4 h-4 text-primary" />
                            <span className="text-[10px] font-black text-white uppercase tracking-widest">Access Raw Perception API</span>
                            <ExternalLink className="w-3 h-3 text-muted-foreground group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// Helper to inject cn if not imported properly (safety)
function cn(...inputs: any[]) {
    return inputs.filter(Boolean).join(' ');
}
