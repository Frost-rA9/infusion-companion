### 关于LiquidLevelDataSet的目录结构

> 推测医院中以中等亮度的情况偏多
>
>  
>
> 观察角度：在输液瓶大致正前方为front，其余为near
>
> 是否有遮挡：视野中是否有杂物
>
> 光纤明暗：开灯等照明良好环境为bright，在未开灯的下午同时照明不佳时为dark，其余为middle
>
> 观察距离：输液瓶清晰可见为near，middle为输液瓶视野中较小，far为输液瓶视野模糊
>
> 液位水平：other > 50 ，其余为分位点

-- front，near ：观察角度

​	-- block，no_block：是否有遮挡

​		-- bright, middle, dark：光纤明暗

​			-- near, middle, far：观察距离

​				-- other, 50,  25, 5, 1：液位水平

