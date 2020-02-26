try:
            for epoch in range(epoch):    
                
                permutation = torch.randperm(x.size()[0])

                for i in range(0,x.size()[0], batch_size):
                    optimizer.zero_grad()           # Forward pass

                    indices = permutation[i:i+batch_size]
                    batch_x, batch_y = x[indices], Y[indices]
                    
                    y_pred = self(batch_x)            # Compute Loss
                
                    loss = criterion(y_pred.squeeze(), batch_y)
                    loss.backward()

                    optimizer.step()
                    print('batch: {} train loss: {}'.format(i, loss.item()))
                
                print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    # Backward pass
                #sys.stdout.write("Epoch : %d , loss : %f \r" % (epoch,loss.item()) )
                #sys.stdout.flush()
        except KeyboardInterrupt:
            print('Training has been stopped at Epoch {}'.format(epoch))
            pass
            # Hyper-paramter, for Backpropagation
           try:
            for epoch in range(epoch):    
                
                permutation = torch.randperm(x.size()[0])

                for i in range(0,x.size()[0], batch_size):
                    optimizer.zero_grad()           # Forward pass

                    indices = permutation[i:i+batch_size]
                    batch_x, batch_y = x[indices], Y[indices]
                    
                    y_pred = self(batch_x)            # Compute Loss
                
                    loss = criterion(y_pred.squeeze(), batch_y)
                    loss.backward()

                    optimizer.step()
                    print('batch: {} train loss: {}'.format(i, loss.item()))
                
                print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    # Backward pass
                #sys.stdout.write("Epoch : %d , loss : %f \r" % (epoch,loss.item()) )
                #sys.stdout.flush()
        except KeyboardInterrupt:
            print('Training has been stopped at Epoch {}'.format(epoch))
            pass
            # Hyper-paramter, for Backpropagation
           
