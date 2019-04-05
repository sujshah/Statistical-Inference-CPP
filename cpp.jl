th = zeros(IT,5)  # th=(psi,mu,tau)
swLabel = zeros(IT+1)  # sandwich label, assign 1 in an iteration if a permutation is done
Aseg = zeros(IT)
Aauxpsi= zeros(IT,2)
AauxpsiJoint=zeros(IT)

Ajoint=0;Asep=0

#th[1,1:2] = drawPsiCentred(y,dx,Tend,I,prior)
#th[1,3:5] =  drawMuTau(y,dx,Tend,I,prior)


for it=1:IT
	th[it,1:2] = drawPsiCentred(y,dx,Tend,I,prior)
	
# joint=1 means: update a pair (psi[i], psi[j])
#if it>1
if it>IT
		# do non-joint updates
		for j=1:2  
 		(th[it,1:2], y, Asep) = drawPsiNoncentred(j,y,th[it,:],dx,dt,I,prior,mcmc,Tend,n,0,dtFull)
 		#println(th[it,1:2],"\n")
 		Aauxpsi[it,j] = Asep
		end
		# do joint update, value of argument j is irrelevant and just set to zero
		(th[it,1:2], y, Ajoint) = drawPsiNoncentred(0,y,th[it,:],dx,dt,I,prior,mcmc,Tend,n,1,dtFull)
		AauxpsiJoint[it] =Ajoint
end

	 th[it,3:5] = drawMuTau(y,dx,Tend,I,prior)
	
	meanA =  0
	count =0
	for i=1:I
	   (y[i,:], A) = updateSegment(vec(y[i,:]),dx[i],dt[i],th[it,:])
		meanA = (A+count*meanA)/(count+1)
		count +=1
	end
	Aseg[it] = meanA
	
	(y, PP, QQ, RR, s) = updateSegmentsMarg(y, dx, dt,Tend, I,prior)

#	if rand()<0.5
#		y=hcat(y[:,2], y[:,1])
#		swLabel[it+1] = abs(swLabel[it]-1)
#	else
#		swLabel[it+1]=swLabel[it]
#	end

	if (mod(it,50)==0)
		println(it, "\t",round(th[it,:],3))
		println(Ajoint, "\n")
		println(Asep, "\n")
	end
end

postPsi=[mean(th[BI:IT,1]) mean(th[BI:IT,2])]
postMu=[mean(th[BI:IT,3]) mean(th[BI:IT,4])]
postSi2=mean(th[BI:IT,5])

print(mean(Aauxpsi),"\n")
print(mean(AauxpsiJoint),"\n")