"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { TrendingUp, TrendingDown, Clock, Package, DollarSign, Target, AlertTriangle, CheckCircle } from "lucide-react"

export default function AnalyticsTab() {
  const monthlyStats = [
    {
      title: "Загальний час друку",
      value: "156г 23хв",
      change: "+12%",
      trend: "up",
      icon: Clock,
    },
    {
      title: "Витрачено пластику",
      value: "2.4 кг",
      change: "+8%",
      trend: "up",
      icon: Package,
    },
    {
      title: "Економія коштів",
      value: "₴1,240",
      change: "+15%",
      trend: "up",
      icon: DollarSign,
    },
    {
      title: "Успішність друку",
      value: "90.4%",
      change: "-2%",
      trend: "down",
      icon: Target,
    },
  ]

  const materialUsage = [
    { name: "PLA Білий", used: 850, total: 1000, color: "#ffffff", cost: 382 },
    { name: "PLA Червоний", used: 950, total: 1000, color: "#dc2626", cost: 427 },
    { name: "ABS Чорний", used: 250, total: 1000, color: "#000000", cost: 130 },
    { name: "PETG Прозорий", used: 700, total: 1000, color: "#f3f4f6", cost: 476 },
  ]

  const topModels = [
    { name: "Кріплення для телефону", prints: 5, time: "11г 15хв", material: "140г", success: 80 },
    { name: "Мініатюра дракона", prints: 3, time: "13г 30хв", material: "135г", success: 100 },
    { name: "Ваза спіральна", prints: 2, time: "12г 40хв", material: "170г", success: 50 },
    { name: "Шестерня для принтера", prints: 1, time: "1г 45хв", material: "22г", success: 100 },
  ]

  const recentIssues = [
    { type: "warning", message: "Друк зупинився через засмічення сопла", date: "2024-01-27" },
    { type: "error", message: "Модель відклеїлась від столу", date: "2024-01-25" },
    { type: "success", message: "Успішно завершено довгий друк (8+ годин)", date: "2024-01-24" },
  ]

  const weeklyActivity = [
    { day: "Пн", hours: 8, prints: 3, success: 100 },
    { day: "Вт", hours: 12, prints: 5, success: 80 },
    { day: "Ср", hours: 6, prints: 2, success: 100 },
    { day: "Чт", hours: 14, prints: 6, success: 83 },
    { day: "Пт", hours: 10, prints: 4, success: 75 },
    { day: "Сб", hours: 16, prints: 7, success: 86 },
    { day: "Нд", hours: 4, prints: 1, success: 100 },
  ]

  const monthlyTrends = [
    { month: "Жов", prints: 45, hours: 120, success: 88 },
    { month: "Лис", prints: 52, hours: 140, success: 85 },
    { month: "Гру", prints: 38, hours: 95, success: 92 },
    { month: "Січ", prints: 47, hours: 156, success: 90 },
  ]

  const maxHours = Math.max(...weeklyActivity.map((d) => d.hours))
  const maxPrints = Math.max(...weeklyActivity.map((d) => d.prints))
  const maxMonthlyHours = Math.max(...monthlyTrends.map((d) => d.hours))
  const maxMonthlyPrints = Math.max(...monthlyTrends.map((d) => d.prints))

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Аналітика</h2>
        <p className="text-muted-foreground">Статистика використання та ефективності</p>
      </div>

      {/* Monthly Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {monthlyStats.map((stat, index) => (
          <Card key={index}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
              <stat.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <div className="flex items-center gap-1 text-xs">
                {stat.trend === "up" ? (
                  <TrendingUp className="h-3 w-3 text-green-500" />
                ) : (
                  <TrendingDown className="h-3 w-3 text-red-500" />
                )}
                <span className={stat.trend === "up" ? "text-green-500" : "text-red-500"}>{stat.change}</span>
                <span className="text-muted-foreground">від минулого місяця</span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Weekly Activity Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Активність за тиждень</CardTitle>
            <CardDescription>Години друку, кількість моделей та успішність</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {weeklyActivity.map((day, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{day.day}</span>
                    <span className="text-muted-foreground">
                      {day.hours}г • {day.prints} моделей • {day.success}%
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <div className="flex-1">
                      <Progress value={(day.hours / maxHours) * 100} className="h-2" />
                    </div>
                    <div className="w-16">
                      <Progress value={(day.prints / maxPrints) * 100} className="h-2" />
                    </div>
                    <div className="w-16">
                      <Progress value={day.success} className="h-2" />
                    </div>
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Години</span>
                    <span>Моделі</span>
                    <span>Успішність</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Monthly Trends */}
        <Card>
          <CardHeader>
            <CardTitle>Тренди по місяцях</CardTitle>
            <CardDescription>Динаміка активності за останні 4 місяці</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {monthlyTrends.map((month, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{month.month}</span>
                    <span className="text-muted-foreground">
                      {month.hours}г • {month.prints} моделей • {month.success}%
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <div className="flex-1">
                      <Progress value={(month.hours / maxMonthlyHours) * 100} className="h-3" />
                    </div>
                    <div className="w-20">
                      <Progress value={(month.prints / maxMonthlyPrints) * 100} className="h-3" />
                    </div>
                    <div className="w-16">
                      <Progress value={month.success} className="h-3" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Material Usage with Cost Analysis */}
        <Card>
          <CardHeader>
            <CardTitle>Використання матеріалів</CardTitle>
            <CardDescription>Витрати пластику та вартість за цей місяць</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {materialUsage.map((material, index) => {
                const percentage = (material.used / material.total) * 100
                return (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full border" style={{ backgroundColor: material.color }} />
                        <span className="text-sm font-medium">{material.name}</span>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-muted-foreground">
                          {material.used}г / {material.total}г
                        </div>
                        <div className="text-xs text-muted-foreground">₴{material.cost}</div>
                      </div>
                    </div>
                    <Progress value={percentage} className="h-2" />
                  </div>
                )
              })}
              <div className="pt-2 border-t">
                <div className="flex justify-between text-sm font-medium">
                  <span>Загальна вартість:</span>
                  <span>₴{materialUsage.reduce((sum, m) => sum + m.cost, 0)}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Top Models with Success Rate */}
        <Card>
          <CardHeader>
            <CardTitle>Найпопулярніші моделі</CardTitle>
            <CardDescription>За кількістю друків та успішністю</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {topModels.map((model, index) => (
                <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                  <div className="flex-1">
                    <div className="font-medium">{model.name}</div>
                    <div className="text-sm text-muted-foreground">
                      {model.time} • {model.material}
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <Badge variant="secondary">{model.prints} друків</Badge>
                    <div
                      className={`text-sm font-medium ${
                        model.success >= 80
                          ? "text-green-600"
                          : model.success >= 60
                            ? "text-yellow-600"
                            : "text-red-600"
                      }`}
                    >
                      {model.success}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Issues */}
      <Card>
        <CardHeader>
          <CardTitle>Останні події</CardTitle>
          <CardDescription>Проблеми та успіхи</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {recentIssues.map((issue, index) => (
              <div key={index} className="flex items-start gap-3 p-3 rounded-lg bg-muted/50">
                <div className="mt-0.5">
                  {issue.type === "warning" && <AlertTriangle className="h-4 w-4 text-yellow-500" />}
                  {issue.type === "error" && <AlertTriangle className="h-4 w-4 text-red-500" />}
                  {issue.type === "success" && <CheckCircle className="h-4 w-4 text-green-500" />}
                </div>
                <div className="flex-1">
                  <div className="text-sm">{issue.message}</div>
                  <div className="text-xs text-muted-foreground mt-1">{issue.date}</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
