import * as React from "react"
import { cn } from "@/lib/utils"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link'
    size?: 'default' | 'sm' | 'lg' | 'icon'
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = 'default', size = 'default', ...props }, ref) => {
        const variants = {
            default: "bg-primary text-black hover:bg-primary/90",
            destructive: "bg-red-600 text-white hover:bg-red-700",
            outline: "border border-sidebar-border bg-transparent hover:bg-sidebar-accent",
            secondary: "bg-sidebar-accent text-white hover:bg-sidebar-accent/80",
            ghost: "hover:bg-sidebar-accent text-white",
            link: "text-primary underline hover:text-primary/80",
        }
        const sizes = {
            default: "h-10 px-4 py-2",
            sm: "h-9 px-3",
            lg: "h-12 px-8 text-base",
            icon: "h-10 w-10",
        }

        return (
            <button
                ref={ref}
                className={cn(
                    "inline-flex items-center justify-center rounded-xl text-sm font-bold transition-all disabled:opacity-50 active:scale-95",
                    variants[variant],
                    sizes[size],
                    className
                )}
                {...props}
            />
        )
    }
)
export { Button }
