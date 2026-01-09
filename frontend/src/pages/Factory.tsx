import { useState, useCallback, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Upload,
    FileImage,
    CheckCircle2,
    AlertTriangle,
    Target,
    Loader2,
    X,
    FolderOpen,
    Archive,
    Search,
    ChevronLeft,
    ChevronRight,
    Filter,
    BarChart3,
    Activity,
    Video,
    VideoOff,
    Layers,
    Eye,
    EyeOff,
    ZoomIn
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';

interface Detection {
    id: string;
    class: string;
    confidence: number;
    bbox: { x: number; y: number; width: number; height: number };
}

interface ResultImage {
    id: string;
    src: string;
    detections: Detection[];
}

const ITEMS_PER_PAGE = 12;

export default function Factory() {
    const [isDragging, setIsDragging] = useState(false);
    const [files, setFiles] = useState<File[]>([]);
    const [results, setResults] = useState<ResultImage[]>([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [processingStatus, setProcessingStatus] = useState('');
    const [currentPage, setCurrentPage] = useState(1);
    const [searchQuery, setSearchQuery] = useState('');
    const [expectedTotal, setExpectedTotal] = useState(0);
    const [zeroShotEnabled, setZeroShotEnabled] = useState(true);
    const [cameraActive, setCameraActive] = useState(false);
    const [liveCounts, setLiveCounts] = useState({ bolt: 0, nut: 0, washer: 0, locatingpin: 0 });
    const [selectedImage, setSelectedImage] = useState<ResultImage | null>(null);
    const [showBboxes, setShowBboxes] = useState(true);
    const [showLabels, setShowLabels] = useState(true);
    const { toast } = useToast();

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const droppedFiles = Array.from(e.dataTransfer.files).filter(file =>
            file.name.toLowerCase().endsWith('.png') ||
            file.name.toLowerCase().endsWith('.jpg') ||
            file.name.toLowerCase().endsWith('.jpeg') ||
            file.name.toLowerCase().endsWith('.zip')
        );

        if (droppedFiles.length !== e.dataTransfer.files.length) {
            toast({
                title: "Invalid File Type",
                description: "Only PNG, JPG images and ZIP archives are supported.",
                variant: "destructive"
            });
        }

        if (droppedFiles.length > 0) {
            setFiles(droppedFiles);
            setResults([]);
            setCurrentPage(1);
        }
    }, [toast]);

    // Real-time WebSocket Log Streaming
    useEffect(() => {
        const ws = new WebSocket('ws://127.0.0.1:8000/ws');

        ws.onmessage = (event) => {
            const msg = event.data;
            if (typeof msg === 'string') {
                if (msg === 'LOG: FACTORY_RESET') {
                    setResults([]);
                    setProcessingStatus('Factory Configured');
                } else if (msg.startsWith('LOG: TOTAL_COUNT:')) {
                    const count = parseInt(msg.split(':')[2]);
                    if (!isNaN(count)) setExpectedTotal(count);
                } else if (msg.startsWith('LOG:')) {
                    setProcessingStatus(msg.replace('LOG: ', ''));
                } else if (msg.includes('PROGRESS')) {
                    setProcessingStatus(msg);
                }
            }
        };

        return () => {
            ws.close();
        };
    }, []);

    // Use polling for real-time result streaming with improved completion logic
    useEffect(() => {
        let pollInterval: NodeJS.Timeout;
        if (isProcessing) {
            pollInterval = setInterval(async () => {
                try {
                    const res = await fetch('http://127.0.0.1:8000/batch_status');
                    const data = await res.json();
                    setResults(data.results);

                    if (!data.is_processing) {
                        setIsProcessing(false);
                        if (data.processed_count >= expectedTotal && expectedTotal > 0) {
                            setProcessingStatus('Batch Analysis Complete');
                            toast({
                                title: "Analysis Finalized",
                                description: `Successfully analyzed ${data.processed_count} images.`,
                            });
                        } else {
                            setProcessingStatus('Analysis Stopped');
                            toast({
                                title: "Analysis Interrupted",
                                description: `Processed ${data.processed_count}/${expectedTotal} images.`,
                                variant: "destructive"
                            });
                        }
                    }
                } catch (err) {
                    console.error("Polling error:", err);
                }
            }, 1000);
        }
        return () => clearInterval(pollInterval);
    }, [isProcessing, expectedTotal, toast]);

    const processBatch = async () => {
        if (files.length === 0) return;
        setIsProcessing(true);
        setResults([]);

        try {
            setProcessingStatus('Clearing Factory Workspace...');
            await fetch('http://127.0.0.1:8000/clear_uploads', { method: 'POST' });

            let actualCount = files.length;
            const zipFile = files.find(f => f.name.endsWith('.zip'));

            if (zipFile) {
                setProcessingStatus(`Streaming ZIP: ${zipFile.name}...`);
                const formData = new FormData();
                formData.append('file', zipFile);

                const res = await fetch(`http://127.0.0.1:8000/upload_zip?augment=${zeroShotEnabled}`, {
                    method: 'POST',
                    body: formData,
                });

                if (res.status === 202) {
                    setProcessingStatus('Upload Accepted. Processing in Background...');
                    return;
                }

                if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
                const zipData = await res.json();
                actualCount = zipData.extracted;
                setExpectedTotal(actualCount);

            } else {
                setProcessingStatus(`Uploading ${files.length} images...`);
                const batchSize = 50;
                for (let i = 0; i < files.length; i += batchSize) {
                    const batch = files.slice(i, i + batchSize);
                    const formData = new FormData();
                    batch.forEach(f => formData.append('files', f));
                    await fetch('http://127.0.0.1:8000/upload_bulk', {
                        method: 'POST',
                        body: formData,
                    });
                    setProcessingStatus(`Transferring: ${Math.min(i + batchSize, files.length)} / ${files.length}`);
                }

                setExpectedTotal(actualCount);
                setProcessingStatus('Activating Neural Engine...');

                await fetch('http://127.0.0.1:8000/predict_all', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ augment: zeroShotEnabled }),
                });
            }

            setProcessingStatus('Processing Real-Time Streams...');

        } catch (error: any) {
            console.error("Factory Process Error:", error);
            toast({
                title: "Factory Failure",
                description: error.message || "Communication with Neural Engine interrupted.",
                variant: "destructive",
            });
            setIsProcessing(false);
        }
    };

    const classColors: Record<string, string> = {
        bolt: 'hsl(187, 100%, 50%)',
        locatingpin: 'hsl(155, 100%, 50%)',
        nut: 'hsl(270, 100%, 60%)',
        washer: 'hsl(38, 100%, 50%)',
    };

    const aggregateCounts = useMemo(() => {
        return results.reduce((acc, res) => {
            res.detections.forEach(det => {
                acc[det.class] = (acc[det.class] || 0) + 1;
            });
            return acc;
        }, {} as Record<string, number>);
    }, [results]);

    const filteredResults = useMemo(() => {
        return results.filter(r => r.id.toLowerCase().includes(searchQuery.toLowerCase()));
    }, [results, searchQuery]);

    const totalPages = Math.ceil(filteredResults.length / ITEMS_PER_PAGE);
    const currentResults = filteredResults.slice(
        (currentPage - 1) * ITEMS_PER_PAGE,
        currentPage * ITEMS_PER_PAGE
    );

    return (
        <div className="space-y-6 max-w-7xl mx-auto pb-12">
            <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight glow-text-cyan flex items-center gap-3 italic uppercase">
                        <Target className="w-8 h-8 text-primary" />
                        AI Factory Engine
                    </h1>
                    <p className="text-muted-foreground mt-1">Real-time processing for SolidWorks assembly streams</p>
                </div>
                {isProcessing && (
                    <div className="flex flex-col items-end gap-1">
                        <span className="text-xs font-mono text-primary animate-pulse">{processingStatus}</span>
                        <div className="h-1.5 w-64 bg-muted rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-primary"
                                initial={{ width: 0 }}
                                animate={{ width: `${(results.length / (expectedTotal || 1)) * 100}%` }}
                            />
                        </div>
                        <span className="text-[10px] text-muted-foreground">{results.length} / {expectedTotal} Frames Analyzed</span>
                    </div>
                )}
            </motion.div>

            <div className="grid grid-cols-1 gap-6">
                {results.length === 0 && !isProcessing && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={cn(
                            "upload-zone min-h-[300px] flex flex-col items-center justify-center cursor-pointer border-2 border-dashed border-sidebar-border transition-all rounded-3xl",
                            isDragging && "border-primary bg-primary/5",
                            files.length > 0 && "border-accent bg-accent/5"
                        )}
                    >
                        <div className="flex flex-col items-center text-center p-8">
                            <div className="flex gap-4 mb-6">
                                <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center">
                                    <Archive className="w-8 h-8 text-primary" />
                                </div>
                                <div className="w-16 h-16 rounded-2xl bg-accent/10 flex items-center justify-center">
                                    <FolderOpen className="w-8 h-8 text-accent" />
                                </div>
                            </div>

                            {files.length === 0 ? (
                                <>
                                    <h3 className="text-xl font-semibold mb-2 uppercase tracking-tight">Drop Project ZIP or Folder</h3>
                                    <p className="text-muted-foreground mb-6 max-w-md">
                                        Isolate SolidWorks parts in milliseconds using YOLO11 real-time vision.
                                    </p>
                                    <div className="flex gap-3">
                                        <Button
                                            variant="outline"
                                            className="rounded-xl border-sidebar-border"
                                            onClick={() => {
                                                const input = document.createElement('input');
                                                input.type = 'file';
                                                (input as any).webkitdirectory = true;
                                                input.onchange = (e) => {
                                                    const files = Array.from((e.target as HTMLInputElement).files || []);
                                                    if (files.length > 0) setFiles(files);
                                                };
                                                input.click();
                                            }}
                                        >
                                            Scan Folder
                                        </Button>
                                        <Button
                                            className="bg-primary text-black font-bold rounded-xl"
                                            onClick={() => {
                                                const input = document.createElement('input');
                                                input.type = 'file';
                                                input.accept = '.zip';
                                                input.onchange = (e) => {
                                                    const file = (e.target as HTMLInputElement).files?.[0];
                                                    if (file) setFiles([file]);
                                                };
                                                input.click();
                                            }}
                                        >
                                            Ingest ZIP
                                        </Button>
                                    </div>
                                </>
                            ) : (
                                <div className="space-y-4">
                                    <div className="flex items-center gap-3 justify-center">
                                        <CheckCircle2 className="w-6 h-6 text-accent" />
                                        <span className="text-lg font-medium">
                                            {files.length === 1 && files[0].name.endsWith('.zip')
                                                ? `Ready to ingest ZIP archive: ${files[0].name}`
                                                : `${files.length} image artifact${files.length !== 1 ? 's' : ''} ready for ingestion`
                                            }
                                        </span>
                                    </div>

                                    <div className="flex items-center gap-4 justify-center glass-card p-4 transition-all hover:border-primary/20">
                                        <div className="flex items-center gap-2">
                                            <Activity className="w-4 h-4 text-accent" />
                                            <span className="text-xs font-bold font-mono uppercase tracking-tighter text-muted-foreground">TTA Augmentation</span>
                                        </div>
                                        <Switch checked={zeroShotEnabled} onCheckedChange={setZeroShotEnabled} />
                                    </div>

                                    <div className="flex gap-3 justify-center">
                                        <Button onClick={processBatch} size="lg" className="bg-primary text-black font-black gap-2 h-14 px-8 rounded-xl shadow-lg shadow-primary/20">
                                            <Target className="w-5 h-5" />
                                            INITIATE FACTORY RUN
                                        </Button>
                                        <Button variant="ghost" onClick={() => setFiles([])} className="text-muted-foreground hover:text-white">
                                            Reset
                                        </Button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}

                {(results.length > 0 || isProcessing) && (
                    <div className="space-y-8">
                        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                            <Card className="glass-card border-sidebar-border bg-sidebar/20">
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                                        <BarChart3 className="w-3 h-3" />
                                        Processed
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <p className="text-3xl font-black font-mono">{results.length}</p>
                                    <p className="text-[10px] text-muted-foreground uppercase tracking-widest mt-1">Real-Time Sync</p>
                                </CardContent>
                            </Card>

                            {Object.entries(classColors).map(([className, color]) => (
                                <Card key={className} className="glass-card border-sidebar-border transition-all hover:border-primary/20" style={{ borderLeft: `3px solid ${color}` }}>
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground italic">{className}</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <p className="text-3xl font-black font-mono" style={{ color: color }}>{aggregateCounts[className] || 0}</p>
                                        <p className="text-[10px] text-muted-foreground uppercase tracking-widest mt-1">Detected</p>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>

                        <div className="flex flex-col md:flex-row items-center justify-between gap-4 glass-card p-6">
                            <div className="relative w-full md:w-96 group">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                                <Input
                                    placeholder="Filter by image name..."
                                    className="pl-10 bg-background/50 border-sidebar-border h-11 rounded-xl focus:ring-primary"
                                    value={searchQuery}
                                    onChange={(e) => { setSearchQuery(e.target.value); setCurrentPage(1); }}
                                />
                            </div>

                            <div className="flex items-center gap-6">
                                <div className="flex items-center gap-4">
                                    <Button variant="outline" size="sm" className="rounded-xl h-9 px-4 border-sidebar-border" disabled={currentPage === 1} onClick={() => setCurrentPage(prev => prev - 1)}>
                                        <ChevronLeft className="w-4 h-4 mr-1" /> PREV
                                    </Button>
                                    <span className="text-[10px] font-black font-mono text-primary uppercase">PAGE {currentPage} / {totalPages || 1}</span>
                                    <Button variant="outline" size="sm" className="rounded-xl h-9 px-4 border-sidebar-border" disabled={currentPage === totalPages || totalPages === 0} onClick={() => setCurrentPage(prev => prev + 1)}>
                                        NEXT <ChevronRight className="w-4 h-4 ml-1" />
                                    </Button>
                                </div>
                                {!isProcessing && (
                                    <Button variant="destructive" size="sm" className="font-black rounded-xl h-9 px-4 uppercase tracking-tighter" onClick={() => { setResults([]); setFiles([]); setCurrentPage(1); setExpectedTotal(0); }}>
                                        CLEAR SESSION
                                    </Button>
                                )}
                            </div>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                            <AnimatePresence>
                                {currentResults.map((result) => (
                                    <motion.div
                                        key={result.id}
                                        initial={{ opacity: 0, scale: 0.9 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        onClick={() => setSelectedImage(result)}
                                        className="glass-card overflow-hidden group border-sidebar-border hover:border-primary/50 transition-all shadow-xl cursor-pointer"
                                    >
                                        <div className="relative aspect-square">
                                            <img src={result.src} className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-700" alt={result.id} />
                                            <div className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 group-hover:opacity-100 transition-all">
                                                <ZoomIn className="w-10 h-10 text-primary" />
                                            </div>
                                            <div className="absolute inset-x-0 bottom-0 p-2 bg-black/90 backdrop-blur-md">
                                                <p className="text-[9px] truncate text-white font-mono font-bold tracking-tighter">{result.id}</p>
                                            </div>
                                            <div className="absolute top-2 right-2 flex flex-col gap-1 items-end">
                                                {result.detections.length === 0 ? (
                                                    <span className="px-2 py-0.5 rounded text-[8px] font-black text-white bg-red-600 leading-none shadow-lg">UNKNOWN</span>
                                                ) : (
                                                    Object.entries(result.detections.reduce((acc, d) => { acc[d.class] = (acc[d.class] || 0) + 1; return acc; }, {} as Record<string, number>)).map(([cls, count]) => (
                                                        <span key={cls} className="px-2 py-0.5 rounded text-[8px] font-black text-black leading-none shadow-md uppercase" style={{ backgroundColor: classColors[cls] }}>{count} {cls.charAt(0)}</span>
                                                    ))
                                                )}
                                            </div>
                                        </div>
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                        </div>
                    </div>
                )}
            </div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mt-12">
                <Card className="bg-sidebar/20 border-sidebar-border overflow-hidden rounded-3xl">
                    <CardHeader className="flex flex-row items-center justify-between p-8 border-b border-sidebar-border/50">
                        <div>
                            <CardTitle className="flex items-center gap-3 text-xl font-black italic uppercase">
                                <Video className="w-6 h-6 text-primary" />
                                Live Industrial Stream
                            </CardTitle>
                            <p className="text-xs text-muted-foreground mt-1">Real-time inference on hardware assembly lines</p>
                        </div>
                        <Button
                            variant={cameraActive ? "destructive" : "default"}
                            className={cn("rounded-xl h-12 px-6 font-black uppercase text-xs tracking-widest shadow-lg", !cameraActive && "bg-primary text-black hover:bg-primary/80")}
                            onClick={async () => {
                                if (cameraActive) {
                                    await fetch('http://localhost:8000/camera/stop', { method: 'POST' });
                                    setCameraActive(false);
                                } else {
                                    setCameraActive(true);
                                }
                            }}
                        >
                            {cameraActive ? <><VideoOff className="w-4 h-4 mr-2" /> DISCONNECT</> : <><Video className="w-4 h-4 mr-2" /> ACTIVATE CAMERA</>}
                        </Button>
                    </CardHeader>
                    <CardContent className="p-8">
                        {cameraActive ? (
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                                <div className="lg:col-span-2">
                                    <div className="relative rounded-[2rem] overflow-hidden bg-black shadow-2xl border border-sidebar-border shadow-primary/10">
                                        <img src="http://localhost:8000/video_feed" alt="Live Camera Feed" className="w-full aspect-video object-contain" />
                                        <div className="absolute top-6 left-6 flex items-center gap-3">
                                            <span className="flex h-3 w-3 relative">
                                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                                                <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                                            </span>
                                            <span className="text-white text-[10px] font-black font-mono bg-black/80 px-3 py-1.5 rounded-full border border-white/10 uppercase tracking-widest">LIVE ANALYTICS</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="space-y-6">
                                    <h3 className="text-sm font-black text-white uppercase italic tracking-wider">Perception Metrics</h3>
                                    <div className="grid grid-cols-2 gap-4">
                                        {Object.entries(liveCounts).map(([cls, count]) => (
                                            <Card key={cls} className="bg-sidebar-accent/50 border-sidebar-border rounded-2xl group transition-all hover:border-primary/30">
                                                <CardContent className="p-6 text-center">
                                                    <div className={cn("text-4xl font-black transition-all", count > 0 ? 'text-primary' : 'opacity-30')}>
                                                        {count}
                                                    </div>
                                                    <div className="text-[10px] text-muted-foreground font-black uppercase tracking-widest mt-2">{cls}</div>
                                                </CardContent>
                                            </Card>
                                        ))}
                                    </div>
                                    <div className="p-6 rounded-2xl bg-primary/5 border border-primary/10">
                                        <p className="text-[10px] text-muted-foreground font-medium leading-relaxed uppercase tracking-tighter">
                                            Neural engine v1.2 optimized for sub-millisecond hardware detection. Target identification active across full FOV.
                                        </p>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="text-center py-20 border-2 border-dashed border-sidebar-border rounded-[2rem] opacity-50 group transition-all hover:opacity-100 cursor-pointer" onClick={() => setCameraActive(true)}>
                                <Video className="w-16 h-16 text-muted-foreground mx-auto mb-6 opacity-30 group-hover:scale-110 transition-transform" />
                                <p className="text-sm font-black text-muted-foreground uppercase tracking-widest">Connect Hardware Stream</p>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </motion.div>

            <AnimatePresence>
                {selectedImage && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 bg-black/98 flex items-center justify-center p-4 backdrop-blur-2xl" onClick={() => setSelectedImage(null)}>
                        <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }} onClick={(e) => e.stopPropagation()} className="relative max-w-6xl w-full glass-card overflow-hidden shadow-2xl border-white/5">
                            <button onClick={() => setSelectedImage(null)} className="absolute top-6 right-6 z-20 p-3 rounded-full bg-black/50 hover:bg-white hover:text-black transition-all border border-white/10"><X className="w-5 h-5" /></button>
                            <button onClick={(e) => { e.stopPropagation(); const currentIndex = filteredResults.findIndex(r => r.id === selectedImage.id); if (currentIndex > 0) setSelectedImage(filteredResults[currentIndex - 1]); }} className="absolute left-6 top-1/2 -translate-y-1/2 z-20 p-4 rounded-full bg-black/50 hover:bg-white hover:text-black transition-all border border-white/10"><ChevronLeft className="w-8 h-8" /></button>
                            <button onClick={(e) => { e.stopPropagation(); const currentIndex = filteredResults.findIndex(r => r.id === selectedImage.id); if (currentIndex < filteredResults.length - 1) setSelectedImage(filteredResults[currentIndex + 1]); }} className="absolute right-6 top-1/2 -translate-y-1/2 z-20 p-4 rounded-full bg-black/50 hover:bg-white hover:text-black transition-all border border-white/10"><ChevronRight className="w-8 h-8" /></button>

                            <div className="relative mx-auto w-fit grid place-items-center">
                                <img
                                    src={selectedImage.src}
                                    alt="Detection result"
                                    className="block h-auto max-h-[75vh] w-auto pointer-events-none select-none"
                                />
                                <div className="absolute inset-0 pointer-events-none">
                                    {showBboxes && selectedImage.detections.map((d) => (
                                        <div
                                            key={d.id}
                                            className="absolute border-2 rounded-sm"
                                            style={{
                                                left: `${d.bbox.x}%`,
                                                top: `${d.bbox.y}%`,
                                                width: `${d.bbox.width}%`,
                                                height: `${d.bbox.height}%`,
                                                borderColor: classColors[d.class],
                                                boxShadow: `0 0 15px -3px ${classColors[d.class]}`,
                                                zIndex: 10
                                            }}
                                        >
                                            {showLabels && (
                                                <span
                                                    className="absolute -top-6 left-0 px-2 py-0.5 text-[9px] font-black rounded shadow-xl uppercase tracking-tighter"
                                                    style={{ backgroundColor: classColors[d.class], color: '#000' }}
                                                >
                                                    {d.class} {(d.confidence * 100).toFixed(0)}%
                                                </span>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>



                            <div className="p-8 bg-sidebar/50 border-t border-white/5">
                                <div className="flex items-center justify-between flex-wrap gap-8">
                                    <div className="flex flex-col">
                                        <span className="text-[10px] font-black text-primary uppercase tracking-widest">Frame Analysis</span>
                                        <span className="text-xl font-black text-white italic uppercase tracking-tighter">{selectedImage.id}</span>
                                    </div>
                                    <div className="flex items-center gap-8">
                                        <div className="flex items-center gap-3">
                                            <Layers className="w-4 h-4 text-primary" />
                                            <span className="text-[10px] font-black uppercase text-muted-foreground mr-2">Boxes</span>
                                            <Switch checked={showBboxes} onCheckedChange={setShowBboxes} />
                                        </div>
                                        <div className="flex items-center gap-3 border-l border-white/10 pl-8">
                                            {showLabels ? <Eye className="w-4 h-4 text-primary" /> : <EyeOff className="w-4 h-4 text-muted-foreground" />}
                                            <span className="text-[10px] font-black uppercase text-muted-foreground mr-2">Labels</span>
                                            <Switch checked={showLabels} onCheckedChange={setShowLabels} />
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-8 border-l border-white/10 pl-8">
                                        {Object.entries(selectedImage.detections.reduce((acc, d) => { acc[d.class] = (acc[d.class] || 0) + 1; return acc; }, {} as Record<string, number>)).map(([cls, count]) => (
                                            <div key={cls} className="text-center group">
                                                <span className="text-2xl font-black" style={{ color: classColors[cls] }}>{count}</span>
                                                <p className="text-[8px] text-muted-foreground font-black uppercase tracking-widest mt-1 opacity-50 group-hover:opacity-100 transition-opacity">{cls}</p>
                                            </div>
                                        ))}

                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
