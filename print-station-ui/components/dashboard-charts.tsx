"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

export default function DashboardCharts() {
  // Sample data for charts
  const weeklyPrintTime = [
    { day: "Пн", hours: 8, prints: 3 },
    { day: "Вт", hours: 12, prints: 5 },
    { day: "Ср", hours: 6, prints: 2 },
    { day: "Чт", hours: 14, prints: 6 },
    { day: "Пт", hours: 10, prints: 4 },
    { day: "Сб", hours: 16, prints: 7 },
    { day: "Нд", hours: 4, prints: 1 },
  ]

  const materialUsage = [
    { name: "PLA", percentage: 45, color: "#3b82f6" },
    { name: "ABS", percentage: 25, color: "#ef4444" },
    { name: "PETG", percentage: 20, color: "#10b981" },
    { name: "TPU", percentage: 10, color: "#f59e0b" },
  ]

  const successRate = [
    { month: "Січ", success: 92, failed: 8 },
    { month: "Лют", success: 88, failed: 12 },
    { month: "Бер", success: 95, failed: 5 },
    { month: "Кві", success: 90, failed: 10 },
  ]

  const maxHours = Math.max(...weeklyPrintTime.map((d) => d.hours))
  const maxPrints = Math.max(...weeklyPrintTime.map((d) => d.prints))

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
      {/* Weekly Activity Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Активність за тиждень</CardTitle>
          <CardDescription>Години друку та кількість моделей</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {weeklyPrintTime.map((day, index) => (
              <div key={index} className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="font-medium">{day.day}</span>
                  <span className="text-muted-foreground">
                    {day.hours}г • {day.prints} моделей
                  </span>
                </div>
                <div className="flex gap-2">
                  <div className="flex-1">
                    <Progress value={(day.hours / maxHours) * 100} className="h-2" />
                  </div>
                  <div className="w-16">
                    <Progress value={(day.prints / maxPrints) * 100} className="h-2 bg-orange-200" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Material Usage Pie Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Використання матеріалів</CardTitle>
          <CardDescription>Розподіл за типами пластику</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="relative w-32 h-32 mx-auto">
              {/* Simple pie chart using conic-gradient */}
              <div
                className="w-full h-full rounded-full"
                style={{
                  background: `conic-gradient(
                    ${materialUsage[0].color} 0% ${materialUsage[0].percentage}%,
                    ${materialUsage[1].color} ${materialUsage[0].percentage}% ${
                      materialUsage[0].percentage + materialUsage[1].percentage
                    }%,
                    ${materialUsage[2].color} ${materialUsage[0].percentage + materialUsage[1].percentage}% ${
                      materialUsage[0].percentage + materialUsage[1].percentage + materialUsage[2].percentage
                    }%,
                    ${materialUsage[3].color} ${
                      materialUsage[0].percentage + materialUsage[1].percentage + materialUsage[2].percentage
                    }% 100%
                  )`,
                }}
              />
              <div className="absolute inset-4 bg-background rounded-full flex items-center justify-center">
                <span className="text-sm font-medium">100%</span>
              </div>
            </div>
            <div className="space-y-2">
              {materialUsage.map((material, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: material.color }} />
                    <span className="text-sm">{material.name}</span>
                  </div>
                  <span className="text-sm font-medium">{material.percentage}%</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Success Rate Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Успішність друку</CardTitle>
          <CardDescription>Відсоток успішних друків по місяцях</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {successRate.map((month, index) => (
              <div key={index} className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="font-medium">{month.month}</span>
                  <span className="text-muted-foreground">{month.success}%</span>
                </div>
                <div className="relative">
                  <Progress value={month.success} className="h-3" />
                  <div
                    className="absolute top-0 right-0 h-3 bg-red-200 rounded-r"
                    style={{ width: `${month.failed}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Успішно: {month.success}%</span>
                  <span>Помилки: {month.failed}%</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
