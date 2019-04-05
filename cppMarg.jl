function updatePQR(yiNew::Array{Int64,2},yi::Array{Int64,2},dxi::Float64, PPold:: Array{Float64,2},QQold,RR)
	li =1/sum(yi)
	liNew = 1/sum(yiNew)
	PP = copy(PPold)
	QQ = copy(QQold)
	#
	PP[1,1] += -li*yi[1]^2 + liNew*yiNew[1]^2
	PP[2,1] += -li*yi[1]*yi[2] +liNew*yiNew[1]*yiNew[2]
	PP[1,2] = PP[2,1]
	PP[2,2] += -li*yi[2]^2 +liNew*yiNew[2]^2
	#

	QQ[1] +=  (yiNew[1]*liNew - yi[1]*li) *dxi
	QQ[2] +=  (yiNew[2]*liNew - yi[2]*li) *dxi

	RR += (liNew-li)* dxi^2

	(PP,QQ,RR)
end


function updateSegmentsMarg(y::Array{Int64,2}, dx::Array{Float64,1}, dt::Array{Float64,1}
								,Tend::Float64,I::Int64,prior::priorspec)

	J = size(y)[2]
	PP=0;QQ=0;RR=0;
	(PP, QQ, RR) = computePQR(y,dx,prior)
	CC = 1/(prior.beta1+RR/2-0.5*dot(QQ,PP\QQ))
	s =sum(y,1)  # colsums
	for i=1:I
		(yiNew, propNew) = proposeSegment(y[i,:])
		(nietnodig, prop) = proposeSegment(yiNew) #SIC
		(PPNew, QQNew, RRNew)=updatePQR(yiNew,y[i,:],dx[i],PP,QQ,RR)

		CCNew = 1/(prior.beta1+RRNew/2-0.5*dot(QQNew,PPNew\QQNew))


		ni = sum(y[i,:])
		niNew = sum(yiNew)
		sNew = s - y[i,:]+yiNew
		
		w= -0.5*log(abs(det(PPNew)/det(PP))) - 0.5*log(niNew/ni) + (niNew-ni)*log(dt[i]/(prior.beta0+Tend))
			+(I/2+prior.alpha1) * log(CCNew/CC)
		for j=1:J
#			w += -lfact(yiNew[j])+lfact(y[i,j]) +lgamma(prior.alpha0+sNew[j])-lgamma(prior.alpha0+s[j])
			w += log(factorial(sNew[j],s[j])*factorial(y[i,j],yiNew[j]))  # only valid if prior.alpha1=1
		end
		
#		println(y[i,:])
#		println(yiNew)
#		println(w)
#		println(i)
		
		A = (prop/propNew) * exp(w)
#		println("A=",A,"\n")
		if rand() <A
#		println("accept")
			y[i,:] = yiNew
			PP=0;QQ=0;RR=0;CC=0;s=0
			PP = PPNew; QQ = QQNew; RR=RRNew; CC=CCNew; s=sNew
		end
	end
	(y, PP, QQ, RR, s)
end



function  proposeSegment(yi::Array{Int64,2})
	yiNew = zeros(2)
	N1=yi[1]
	N2=yi[2]
	prob=0 # probability of the chosen proposal
	if (N1==0) & (N2==1)
		U = 3*rand()
		prob = 1/3
		if U<1
			yiNew = [0 2]
		elseif 1<U<2
			yiNew = [1 1]
		else
			yiNew = [1 0]
		end
		return((yiNew, prob))
	end
	#--
	if (N2==0) & (N1==1)
		U = 3*rand()
		prob = 1/3
		if U<1
			yiNew = [0 1]
		elseif 1<U<2
			yiNew = [1 1]
		else
			yiNew = [2 0]
		end
		return((yiNew, prob))
	end
	#--
	if (N1==0) & (N2>1)
		U = 4*rand()
		prob = .25
		if U<1
			yiNew = [0 N2-1]
		elseif 1<U<2
			yiNew = [0 N2+1]
		elseif 2 < U< 3
			yiNew= [1 N2]
		else 3 < U < 4
			yiNew =[1 N2-1]
		end
		return((yiNew, prob))
	end

	if (N2==0) & (N1>1)
		U = 4*rand()
		prob = 0.25
		if U<1
			yiNew = [N1-1 0]
		elseif 1<U<2
			yiNew = [N1+1 0]
		elseif 2 < U< 3
			yiNew= [N1-1 1]
		else 3 < U < 4
			yiNew =[N1 1]
		end
		return((yiNew, prob))
	end

	if (N1>0) & (N2>0)
		U = 6*rand()
		prob = 1/6
#		println(prob)
		if U<1
			yiNew = [N1+1 N2]
		elseif 1< U < 2
			yiNew = [N1-1 N2]
		elseif 2 < U < 3
			yiNew = [N1 N2+1]
		elseif 3< U< 4
			yiNew = [N1 N2-1]
		elseif 4< U< 5
			yiNew = [N1-1 N2+1]
#		elseif 0.625< U<0.75
#			yiNew = [N1+1 N2+1]
#		elseif 0.75< U<0.875
#			yiNew = [N1-1 N2-1]
		else
			yiNew = [N1+1 N2-1]
		end
		return((yiNew, prob))
	end
end


