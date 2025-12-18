"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Play, Pause, Square, ArrowUp, ArrowDown, Trash2, Clock, Package, Calendar } from "lucide-react"

export default function QueueTab() {
  const [queue, setQueue] = useState([
    {
      id: 1,
      name: "Мініатюра дракона",
      status: "printing",
      progress: 67,
      printTime: "4г 30хв",
      timeLeft: "2г 15хв",
      material: "PLA Червоний",
      materialUsed: 45,
      priority: "high",
      startTime: "14:30",
    },
    {
      id: 2,
      name: "Кріплення для телефону",
      status: "queued",
      progress: 0,
      printTime: "2г 15хв",
      timeLeft: "2г 15хв",
      material: "PLA Білий",
      materialUsed: 28,
      priority: "medium",
      startTime: "17:45",
    },
    {
      id: 3,
      name: "Шестерня для принтера",
      status: "queued",
      progress: 0,
      printTime: "1г 45хв",
      timeLeft: "1г 45хв",
      material: "ABS Чорний",
      materialUsed: 22,
      priority: "high",
      startTime: "20:00",
    },
    {
      id: 4,
      name: "Ваза спіральна",
      status: "queued",
      progress: 0,
      printTime: "6г 20хв",
      timeLeft: "6г 20хв",
      material: "PETG Прозорий",
      materialUsed: 85,
      priority: "low",
      startTime: "21:45",
    },
  ])

  const moveUp = (id: number) => {
    const index = queue.findIndex((item) => item.id === id)
    if (index > 0) {
      const newQueue = [...queue]
      ;[newQueue[index], newQueue[index - 1]] = [newQueue[index - 1], newQueue[index]]
      setQueue(newQueue)
    }
  }

  const moveDown = (id: number) => {
    const index = queue.findIndex((item) => item.id === id)
    if (index < queue.length - 1) {
      const newQueue = [...queue]
      ;[newQueue[index], newQueue[index + 1]] = [newQueue[index + 1], newQueue[index]]
      setQueue(newQueue)
    }
  }

  const removeFromQueue = (id: number) => {
    setQueue(queue.filter((item) => item.id !== id))
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "printing":
        return <Badge className="bg-blue-500">Друкується</Badge>
      case "queued":
        return <Badge variant="secondary">В черзі</Badge>
      case "paused":
        return <Badge variant="outline">Пауза</Badge>
      default:
        return <Badge variant="outline">{status}</Badge>
    }
  }

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case "high":
        return <Badge variant="destructive">Високий</Badge>
      case "medium":
        return <Badge variant="secondary">Середній</Badge>
      case "low":
        return <Badge variant="outline">Низький</Badge>
      default:
        return <Badge variant="outline">{priority}</Badge>
    }
  }

  const totalTime = queue.reduce((acc, item) => {
    const [hours, minutes] = item.timeLeft.split("г ").map((t) => Number.parseInt(t.replace("хв", "")))
    return acc + hours * 60 + (minutes || 0)
  }, 0)

  const totalMaterial = queue.reduce((acc, item) => acc + item.materialUsed, 0)

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Черга друку</h2>
          <p className="text-muted-foreground">Планування та управління чергою друку</p>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Загальний час</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {Math.floor(totalTime / 60)}г {totalTime % 60}хв
            </div>
            <p className="text-xs text-muted-foreground">{queue.length} моделей в черзі</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Потрібно пластику</CardTitle>
            <Package className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalMaterial}г</div>
            <p className="text-xs text-muted-foreground">Для всієї черги</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Завершення</CardTitle>
            <Calendar className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">04:05</div>
            <p className="text-xs text-muted-foreground">Завтра ранок</p>
          </CardContent>
        </Card>
      </div>

      {/* Queue Items */}
      <div className="space-y-4">
        {queue.map((item, index) => (
          <Card key={item.id} className={item.status === "printing" ? "border-blue-500" : ""}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="text-2xl font-bold text-muted-foreground">#{index + 1}</div>
                  <div>
                    <CardTitle className="text-lg">{item.name}</CardTitle>
                    <CardDescription>
                      Початок: {item.startTime} • {item.material}
                    </CardDescription>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {getPriorityBadge(item.priority)}
                  {getStatusBadge(item.status)}
                </div>
              </div>
            </CardHeader>

            <CardContent>
              <div className="space-y-4">
                {item.status === "printing" && (
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Прогрес</span>
                      <span>
                        {item.progress}% • Залишилось: {item.timeLeft}
                      </span>
                    </div>
                    <Progress value={item.progress} className="h-2" />
                  </div>
                )}

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Час друку</div>
                    <div className="font-medium">{item.printTime}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Пластик</div>
                    <div className="font-medium">{item.materialUsed}г</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Матеріал</div>
                    <div className="font-medium">{item.material}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Початок</div>
                    <div className="font-medium">{item.startTime}</div>
                  </div>
                </div>

                <div className="flex justify-between items-center pt-2">
                  <div className="flex gap-2">
                    {item.status === "printing" ? (
                      <>
                        <Button size="sm" variant="outline">
                          <Pause className="h-4 w-4" />
                        </Button>
                        <Button size="sm" variant="outline">
                          <Square className="h-4 w-4" />
                        </Button>
                      </>
                    ) : (
                      <Button size="sm">
                        <Play className="h-4 w-4 mr-2" />
                        Запустити
                      </Button>
                    )}
                  </div>

                  {item.status !== "printing" && (
                    <div className="flex gap-1">
                      <Button size="sm" variant="ghost" onClick={() => moveUp(item.id)} disabled={index === 0}>
                        <ArrowUp className="h-4 w-4" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => moveDown(item.id)}
                        disabled={index === queue.length - 1}
                      >
                        <ArrowDown className="h-4 w-4" />
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => removeFromQueue(item.id)}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
