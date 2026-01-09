import { motion } from 'framer-motion';
import {
    Rocket,
    Cpu,
    Target,
    Zap,
    ShieldCheck,
    Layers,
    ChevronRight,
    Database,
    Search,
    CheckCircle2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useNavigate } from 'react-router-dom';

const capabilities = [
    {
        title: 'Precision Bolt Indexing',
        desc: 'Automated identification and localization of industrial fastener units with sub-millimeter accuracy.',
        icon: Target,
        color: 'hsl(187, 100%, 50%)'
    },
    {
        title: 'Locating Pin Perception',
        desc: 'Real-time orientation and coordinate mapping for assembly alignment hardware.',
        icon: Layers,
        color: 'hsl(155, 100%, 50%)'
    },
    {
        title: 'Torque-Component Vision',
        desc: 'High-speed detection of hex nuts and securing hardware in complex industrial environments.',
        icon: ShieldCheck,
        color: 'hsl(270, 100%, 60%)'
    },
    {
        title: 'Washer Interface Analysis',
        desc: 'Structural verification of load-bearing interface components during assembly workflows.',
        icon: CheckCircle2,
        color: 'hsl(38, 100%, 50%)'
    }
];

const specs = [
    { label: 'Inference Latency', value: '25ms', icon: Zap, sub: 'Edge Optimized' },
    { label: 'Dataset Magnitude', value: '9,558', icon: Database, sub: 'Synthetic Artifacts' },
    { label: 'Model Architecture', value: 'YOLO11', icon: Cpu, sub: 'Neural Engine v1.2' },
    { label: 'Perception Clarity', value: '88.4%', icon: Rocket, sub: 'mAP@50 Accuracy' }
];

export default function Dashboard() {
    const navigate = useNavigate();

    return (
        <div className="space-y-12 pb-12">
            {/* Hero Section */}
            <section className="relative overflow-hidden rounded-[2rem] bg-gradient-to-br from-sidebar to-background border border-sidebar-border p-8 md:p-16">
                <div className="absolute top-0 right-0 w-1/2 h-full opacity-10 pointer-events-none">
                    <Search className="w-full h-full text-primary animate-pulse" />
                </div>

                <motion.div
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="relative z-10 max-w-2xl"
                >
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-bold mb-6">
                        <Rocket className="w-3 h-3" />
                        <span>PROJECT SENTINEL • INDUSTRIAL AI SOLUTIONS</span>
                    </div>

                    <h1 className="text-5xl md:text-7xl font-black tracking-tighter mb-6 leading-[0.95] text-white">
                        NEXT-GEN <br />
                        <span className="glow-text-cyan italic">NEURAL VISION</span>
                    </h1>

                    <p className="text-xl text-muted-foreground mb-8 leading-relaxed max-w-lg">
                        A high-performance deep learning ecosystem designed to automate SolidWorks assembly inspection through real-time artifact perception and localization.
                    </p>

                    <div className="flex flex-wrap gap-4">
                        <Button
                            size="lg"
                            className="bg-primary hover:bg-primary/80 text-black font-black gap-2 h-14 px-8 rounded-xl"
                            onClick={() => navigate('/factory')}
                        >
                            LAUNCH AI FACTORY
                            <ChevronRight className="w-5 h-5" />
                        </Button>
                        <Button
                            size="lg"
                            variant="outline"
                            className="border-sidebar-border hover:bg-white/5 font-bold h-14 px-8 rounded-xl"
                            onClick={() => navigate('/live-camera')}
                        >
                            LIVE PERCEPTION
                        </Button>
                    </div>
                </motion.div>
            </section>

            {/* Technical Specs */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
                {specs.map((spec, i) => (
                    <motion.div
                        key={spec.label}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="glass-card p-6 border-sidebar-border hover:border-primary/30 transition-all group"
                    >
                        <spec.icon className="w-5 h-5 text-primary mb-4 group-hover:scale-110 transition-transform" />
                        <p className="text-3xl font-black text-white font-mono">{spec.value}</p>
                        <p className="text-xs font-bold text-muted-foreground uppercase tracking-widest mt-1">{spec.label}</p>
                        <p className="text-[10px] text-primary/60 font-mono mt-1">{spec.sub}</p>
                    </motion.div>
                ))}
            </div>

            {/* Capabilities Section */}
            <section className="space-y-8">
                <div className="flex items-center justify-between">
                    <h2 className="text-2xl font-black tracking-tight text-white uppercase italic">Model Capabilities</h2>
                    <div className="h-px flex-1 mx-8 bg-sidebar-border" />
                    <div className="text-[10px] font-mono text-muted-foreground uppercase tracking-[0.3em]">Neural Interface v1.0.4</div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {capabilities.map((cap, i) => (
                        <motion.div
                            key={cap.title}
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: 0.4 + i * 0.1 }}
                            className="glass-card p-8 group hover:border-sidebar-border/80 transition-all flex gap-6"
                        >
                            <div
                                className="w-16 h-16 shrink-0 rounded-2xl flex items-center justify-center border border-white/5"
                                style={{ backgroundColor: `${cap.color}10`, boxShadow: `0 10px 30px -10px ${cap.color}30` }}
                            >
                                <cap.icon className="w-8 h-8" style={{ color: cap.color }} />
                            </div>
                            <div>
                                <h3 className="text-xl font-black text-white mb-2 group-hover:text-primary transition-colors uppercase tracking-tight italic">
                                    {cap.title}
                                </h3>
                                <p className="text-muted-foreground leading-relaxed text-sm">
                                    {cap.desc}
                                </p>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </section>

            {/* Industrial Workflow Footer */}
            <motion.section
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                className="glass-card p-12 text-center bg-gradient-to-t from-primary/5 to-transparent border-primary/10 overflow-hidden relative"
            >
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none" />
                <h3 className="text-3xl font-black text-white mb-4 italic uppercase tracking-tighter">Automate the Industrial Loop</h3>
                <p className="text-muted-foreground max-w-2xl mx-auto mb-8">
                    The Neural Engine seamlessly integrates into existing factory pipelines, providing instantaneous verification for hardware components and reducing manual inspection overhead by up to 94%.
                </p>
                <div className="flex justify-center gap-12 grayscale opacity-50">
                    <div className="flex items-center gap-2 font-black text-white italic"><Cpu className="w-5 h-5" /> SOLIDWORKS</div>
                    <div className="flex items-center gap-2 font-black text-white italic"><Target className="w-5 h-5" /> NVIDIA</div>
                    <div className="flex items-center gap-2 font-black text-white italic"><Rocket className="w-5 h-5" /> PYTORCH</div>
                </div>
            </motion.section>
        </div>
    );
}
