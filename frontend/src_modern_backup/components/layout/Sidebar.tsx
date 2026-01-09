import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
    LayoutDashboard,

    Factory,
    Download,
    Video,
    Library,
    Settings,
    Shield,
    Menu,
    X
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { useState } from 'react';

const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/factory', label: 'Factory Engine', icon: Factory },
    { path: '/live-camera', label: 'Live Perception', icon: Video },
    { path: '/export', label: 'Export Hub', icon: Download },
];

export function Sidebar() {
    const navigate = useNavigate();
    const location = useLocation();
    const [collapsed, setCollapsed] = useState(false);

    return (
        <div className={cn(
            "h-screen bg-sidebar flex flex-col border-r border-sidebar-border transition-all duration-500 relative",
            collapsed ? "w-20" : "w-72"
        )}>
            <div className="p-6 mb-8 flex items-center justify-between">
                {!collapsed && (
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center shadow-lg shadow-primary/20">
                            <Shield className="w-6 h-6 text-black" />
                        </div>
                        <div>
                            <h1 className="text-lg font-black tracking-tighter text-white leading-none">SENTINEL</h1>
                            <span className="text-[10px] font-black text-primary/60 tracking-widest uppercase">AI Engine</span>
                        </div>
                    </div>
                )}
                <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setCollapsed(!collapsed)}
                    className="text-muted-foreground hover:bg-sidebar-accent"
                >
                    {collapsed ? <Menu className="w-5 h-5" /> : <X className="w-5 h-5" />}
                </Button>
            </div>

            <nav className="flex-1 px-4 space-y-2">
                {navItems.map((item) => {
                    const isActive = location.pathname === item.path;
                    return (
                        <button
                            key={item.path}
                            onClick={() => navigate(item.path)}
                            className={cn(
                                "w-full flex items-center gap-4 px-4 py-4 rounded-2xl transition-all duration-300 group relative",
                                isActive
                                    ? "bg-primary/10 text-primary"
                                    : "text-muted-foreground hover:bg-sidebar-accent hover:text-white"
                            )}
                        >
                            <item.icon className={cn(
                                "w-5 h-5 transition-transform duration-300",
                                isActive ? "scale-110" : "group-hover:scale-110"
                            )} />
                            {!collapsed && (
                                <span className="text-sm font-bold uppercase tracking-widest italic">{item.label}</span>
                            )}
                            {isActive && (
                                <motion.div
                                    layoutId="active-indicator"
                                    className="absolute left-0 w-1 h-8 bg-primary rounded-full"
                                />
                            )}
                        </button>
                    );
                })}
            </nav>

            <div className="p-6 mt-auto">
                <div className={cn(
                    "rounded-2xl bg-sidebar-accent/50 p-4 border border-sidebar-border",
                    collapsed ? "hidden" : "block"
                )}>
                    <div className="flex items-center gap-2 mb-2">
                        <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
                        <span className="text-[10px] font-black text-accent uppercase tracking-widest">Active Status</span>
                    </div>
                    <p className="text-[10px] text-muted-foreground uppercase font-bold tracking-tight">V1.0.4-PRO-ALPHA</p>
                </div>
            </div>
        </div>
    );
}
